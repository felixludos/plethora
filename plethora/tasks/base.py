
import torch

from omnibelt import get_printer

from ..framework.features import Seeded
from ..framework.base import TensorDict

prt = get_printer(__file__)

class ResultsContainer(TensorDict):
	def __init__(self, dataset, **kwargs):
		super().__init__(**kwargs)
		self.dataset = dataset


	def set_batch(self, batch):
		self.batch = batch



class Task(Seeded): # TODO: specify random seed for reproducibility
	def __init__(self, dataset=None, slim=False, online=False, score_key=None, **kwargs):
		super().__init__(**kwargs)
		self.dataset = dataset
		self._slim = slim
		self._online = online
		self._score_key = score_key


	@staticmethod
	def score_names():
		return ['score']


	@staticmethod
	def create_results_container(dataset=None, **kwargs):
		return ResultsContainer(dataset)
	
	
	@classmethod
	def prepare(cls, dataset=None, **kwargs):
		return cls.create_results_container(dataset=dataset, **kwargs)
		
		
	@staticmethod
	def run(info, slim=False, online=False, seed=None, gen=None):
		raise NotImplementedError


	@classmethod
	def select_results(cls, inp, out, slim=False, online=False, score_key=None):
		scores = set(cls.score_names())
		for key in list(out.keys()):
			if slim and key not in scores:
				del out[key]
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
		if slim is None:
			slim = self._slim
		if online is None:
			online = self._online
		if seed is not None:
			gen = torch.Generator().manual_seed(seed)
		if gen is None:
			gen = self.gen
		return self._compute(info=info, slim=slim, online=online, gen=gen, seed=seed, **kwargs)


	def _compute(self, info=None, slim=False, online=False, gen=None, seed=None, **kwargs):
		if info is None:
			info = self.prepare(dataset=self.dataset, **kwargs)
		out = self.run(info, slim=slim, online=online, gen=gen, seed=seed)
		return self.select_results(info, out, slim=slim, online=online)
		


# class IterativeResultsContainer(ResultsContainer):
# 	# def __init__(self, infinite=False, sample_format=None,
# 	#              drop_last=None, batch_size=None, **kwargs):
# 	# 	super().__init__(**kwargs)
# 	# 	self.batch = None
# 	# 	self._dataset_itr = None
# 	#
# 	# 	self.epochs_complete = 0
# 	# 	self._infinite = infinite
# 	# 	self._sample_format = sample_format
# 	# 	self._drop_last = drop_last
# 	# 	self._batch_size = batch_size
# 	#
# 	#
# 	# def _create_dataset_iterator(self):
# 	# 	return self.dataset.get_iterator(infinite=self._infinite, sample_format=self._sample_format,
# 	# 	                                 drop_last=self._drop_last, batch_size=self._batch_size)
# 	#
# 	#
# 	# def next_batch(self):
# 	# 	if self._dataset_itr is None:
# 	# 		self._dataset_itr = self._create_dataset_iterator()
# 	# 	batch = next(self._dataset_itr)
# 	# 	self.set_batch(batch)
# 	# 	return batch
#
#
# 	def set_batch(self, batch):
# 		self.batch = batch



class BatchedTask(Task):
	def __init__(self, sample_format=None, drop_last=None, batch_size=None, load_missing=True, **kwargs):
		super().__init__(**kwargs)
		self._sample_format = sample_format
		self._batch_size = batch_size
		self._drop_last = drop_last
		self._load_missing = load_missing
	
	
	def _compute(self, info=None, batch=None, slim=False, online=False, gen=None, seed=None, **kwargs):
		if info is None:
			info = self.prepare(dataset=self.dataset, **kwargs)
		out = self.run(info, batch=batch, load_missing=self._load_missing, sample_format=self._sample_format,
		               drop_last=self._drop_last, batch_size=self._batch_size,
		               slim=slim, online=online, gen=gen, seed=seed)
		return self.select_results(info, out, slim=slim, online=online, score_key=self._score_key)
	
	
	@classmethod
	def run(cls, info, batch=None, load_missing=True,
	        sample_format=None, drop_last=None, batch_size=None,
	        slim=False, online=False, seed=None, gen=None):
		if batch is None:
			for batch in info.dataset.get_iterator(sample_format=sample_format, drop_last=drop_last,
			                                       batch_size=batch_size):
				info.set_batch(batch)
				info = cls.run_step(batch, info, slim=slim, online=online, seed=seed, gen=gen)
		else:
			if sample_format is not None:
				batch = info.dataset.change_sample_format(batch, sample_format=sample_format, get_missing=load_missing)
			info.set_batch(batch)
			info = cls.run_step(batch, info, slim=slim, online=online, gen=gen, seed=seed)
		return cls.aggregate(info, slim=slim, online=online, seed=seed, gen=gen)


	@staticmethod
	def run_step(batch, info, slim=False, online=False, seed=None, gen=None):
		raise NotImplementedError

	
	@staticmethod
	def aggregate(info, slim=False, online=False, seed=None, gen=None):
		return info
	
	







