
import torch

from omnibelt import get_printer, agnosticmethod

from ..framework.features import Seeded
from ..framework.base import Container

prt = get_printer(__file__)


class Task(Seeded): # TODO: specify random seed for reproducibility
	def __init__(self, dataset=None, slim=False, online=False, score_key=None, **kwargs):
		super().__init__(**kwargs)
		self.dataset = dataset
		self._slim = slim
		self._online = online
		self.score_key = score_key


	@agnosticmethod
	def score_names(self):
		return [] if self.score_key is None else ['score']

	
	class ResultsContainer(Seeded, Container):
		pass
		# def __init__(self, dataset, slim=False, online=False, **kwargs):
		# 	super().__init__(**kwargs)
		# 	self.dataset = dataset
		# 	self.online = online
		# 	self.slim = slim

	
	@classmethod
	def create_results_container(cls, dataset=None, slim=False, online=False, **kwargs):
		return cls.ResultsContainer(dataset=dataset, slim=slim, online=online, **kwargs)
	
	
	@agnosticmethod
	def prepare(self, info=None, slim=None, online=None, gen=None, seed=None, **kwargs):
		if slim is None:
			slim = getattr(self, '_slim', False)
		if online is None:
			online = getattr(self, '_online', False)
		if gen is None:
			gen = getattr(self, 'gen', None)
		if seed is not None or gen is None:
			gen = self.create_rng(gen=gen, seed=seed)
		if info is None:
			info = self.create_results_container(seed=seed, gen=gen, **kwargs)
		
		info.dataset = self.dataset
		info.slim = slim
		info.online = online
		return info
		
		
	@staticmethod
	def run(info):
		raise NotImplementedError

	
	score_key = None
	@agnosticmethod
	def select_results(self, inp, out, score_key=None):
		if score_key is None:
			score_key = self.score_key
		# scores = set(self.score_names())
		# for key in list(out.keys()):
		# 	if inp.slim and key not in scores:
		# 		del out[key]
		if 'score' not in out and score_key is not None and score_key in out:
			out['score'] = out[score_key]
		return out


	def compute(self, slim=None, online=None, seed=None, gen=None, info=None, **kwargs):
		'''
		:param slim: use little space (in results)
		:param online: use little time (in computing results)
		:param seed: random seed
		:param gen: torch.Generator instance (defaults to self.gen)
		:param info: results container (created using self._create_results_container)
		:return: results container
		'''
		if info is None:
			info = self.prepare(slim=slim, online=online, seed=seed, gen=gen, **kwargs)
		out = self.run(info)
		return self.select_results(info, out, score_key=self.score_key)
		


class BatchedTask(Task):
	def __init__(self, num_samples=None, batch_size=None, force_batch_size=None, hard_sample_limit=None,
	             pbar=None, **kwargs):
		super().__init__(**kwargs)
		self._num_samples = num_samples
		self._batch_size = batch_size
		self._force_batch_size = force_batch_size
		self._hard_sample_limit = hard_sample_limit
		self._pbar = pbar
	
	
	class ResultsContainer(Task.ResultsContainer):
		def set_batch(self, batch):
			self.batch = batch
		
		
		def _find_missing(self, key, **kwargs):
			if self.batch is not None:
				self[key] = self.batch.get(key) # load and cache
				return self[key]
			return super()._find_missing(key)
		
	
	def compute(self, info=None, batch=None, **kwargs):
		info = self.prepare(info=info, **kwargs)
		out = self.run(info=info, batch=batch, num_samples=self._num_samples, batch_size=self._batch_size,
		               force_batch_size=self._force_batch_size, hard_sample_limit=self._hard_sample_limit,
		               pbar=self._pbar)
		return self.select_results(info, out, score_key=self.score_key)
	
	
	def run(self, info, batch=None,
	        num_samples=None, batch_size=None, force_batch_size=None, hard_sample_limit=None, pbar=None):
		if batch is None:
			itr = info.dataset.get_iterator(num_samples=num_samples, batch_size=batch_size, pbar=pbar, gen=info.gen,
			                                force_batch_size=force_batch_size, hard_limit=hard_sample_limit)
			for batch in itr:
				info = self.run_step(batch=batch, info=info)
		else:
			info = self.run_step(batch=batch, info=info)
		return self.aggregate(info=info)


	@staticmethod
	def run_step(batch, info):
		raise NotImplementedError

	
	@staticmethod
	def aggregate(info):
		return info
	
	

class SimpleEvaluationTask(BatchedTask):
	score_key = 'mean'
	
	
	class ResultsContainer(BatchedTask.ResultsContainer):
		def __init__(self, **kwargs):
			super().__init__(**kwargs)
			self.criteria = []
		
		
		def accumulate(self, criteria):
			self.criteria.append(criteria)
		
		
		def aggregate(self):
			return torch.cat(self.criteria)


	@classmethod
	def score_names(cls):
		return ['mean', 'std', 'min', 'max', *super().score_names()]

	
	@classmethod
	def run_step(cls, batch, info, *, comparison_key='comparison'):
		info.clear()
		info.set_batch(batch)
		cls._eval_step(info, comparison_key=comparison_key)
		if comparison_key in info:
			info.accumulate(comparison_key)
		return info
	
	
	@staticmethod
	def _eval_step(info, *, comparison_key='comparison'):
		raise NotImplementedError
	

	@staticmethod
	def aggregate(info):
		info = super().aggregate(info)

		criteria = info.aggregate()

		info.update({
			'criteria': criteria,
			'mean': criteria.mean().item(),
			'max': criteria.max().item(),
			'min': criteria.min().item(),
			'std': criteria.std().item(),
		})
		if info.slim:
			del info['criteria']
		return info






