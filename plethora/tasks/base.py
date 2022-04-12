
import torch

from omnibelt import get_printer, agnosticmethod, unspecified_argument

from ..framework.features import Seeded, Prepared
from ..framework.base import Container
from ..framework.models import Computable

prt = get_printer(__file__)


class Task(Computable, Prepared, Seeded): # TODO: specify random seed for reproducibility
	_slim = False
	_online = False

	def __init__(self, dataset=None, slim=None, online=None, **kwargs):
		super().__init__(**kwargs)
		self.dataset = dataset
		if slim is not None:
			self._slim = slim
		if online is not None:
			self._online = online


	@agnosticmethod
	def score_names(self):
		return set() if self.score_key is None else {'score'}


	@agnosticmethod
	def heavy_results(self):
		return set()

	
	class ResultsContainer(Computable.ResultsContainer):
		def __init__(self, dataset=None, slim=False, online=False, **kwargs):
			super().__init__(**kwargs)
			self.dataset = dataset
			self.online = online
			self.slim = slim


		@property
		def source(self):
			return self._source if self._source is None else self.dataset
		@source.setter
		def source(self, source):
			self._source = source


	def _prepare(self, *args, **kwargs):
		pass


	@agnosticmethod
	def create_results_container(self, source=None, dataset=None, slim=None, online=None,
	                             gen=None, seed=unspecified_argument, **kwargs):
		if dataset is None:
			dataset = getattr(self, 'dataset', None)
		if slim is None:
			slim = self._slim
		if online is None:
			online = self._online
		if online is None: # if a source is provided, by default, that makes it "online"
			online = source is not None
		if seed is unspecified_argument:
			seed = self._seed
		return super().create_results_container(source=source, dataset=dataset, slim=slim, online=online,
		                                        seed=seed, **kwargs)
		
		
	@staticmethod
	def run(info, **kwargs):
		raise NotImplementedError


	@agnosticmethod
	def select_results(self, inp, out):
		heavy = self.heavy_results()
		if getattr(out, 'slim', False):
			for key in heavy:
				if key in out:
					del out[key]
		return out


	@agnosticmethod
	def compute(self, source=None, **kwargs):
		'''
		:param slim: use little space (in results)
		:param online: use little time (in computing results)
		:param seed: random seed
		:param gen: torch.Generator instance (defaults to self.gen)
		:param info: results container (created using self._create_results_container)
		:return: results container
		'''
		self.prepare()
		return super().compute(source=source, **kwargs)


	@agnosticmethod
	def _compute(self, info, **kwargs):
		out = self.run(info, **kwargs)
		return self.select_results(info, out)

		


class BatchedTask(Task):
	num_samples = None
	batch_size = None
	force_batch_size = None
	hard_sample_limit = None
	pbar = None

	def __init__(self, num_samples=unspecified_argument, batch_size=unspecified_argument,
	             force_batch_size=unspecified_argument, hard_sample_limit=unspecified_argument,
	             pbar=unspecified_argument, **kwargs):
		super().__init__(**kwargs)
		if num_samples is not unspecified_argument:
			self.num_samples = num_samples
		if batch_size is not unspecified_argument:
			self.batch_size = batch_size
		if force_batch_size is not unspecified_argument:
			self.force_batch_size = force_batch_size
		if hard_sample_limit is not unspecified_argument:
			self.hard_sample_limit = hard_sample_limit
		if pbar is not unspecified_argument:
			self.pbar = pbar


	class ResultsContainer(Task.ResultsContainer):
		def new_batch(self, batch):
			self.clear()
			self.source = batch


		@property
		def source(self): # prevent source from defaulting to
			return self._source
		@source.setter
		def source(self, source):
			self._source = source


	@agnosticmethod
	def loop(self, info):
		if info.source is not None:
			info.new_batch(info.source)
			yield info
			return

		itr = info.dataset.get_iterator(num_samples=self.num_samples, batch_size=self.batch_size, pbar=self.pbar,
		                                force_batch_size=self.force_batch_size, hard_limit=self.hard_sample_limit,
		                                gen=info.gen)
		for batch in itr:
			info.new_batch(batch)
			yield info


	@agnosticmethod
	def run(self, info):
		for batch in self.loop(info):
			self.run_step(batch)
		return self.aggregate(info)


	@staticmethod
	def run_step(info):
		raise NotImplementedError

	
	@staticmethod
	def aggregate(info):
		return info
	


class Cumulative(BatchedTask.ResultsContainer):
	_auto_cumulative_keys = None

	def __init__(self, auto_cumulative_keys=unspecified_argument, **kwargs):
		super().__init__(**kwargs)
		if auto_cumulative_keys is not unspecified_argument:
			self._auto_cumulative_keys = auto_cumulative_keys
		self._cumulatives = {}


	def accumulate(self, key, val):
		if key not in self._cumulatives:
			self._cumulatives[key] = []
		self._cumulatives[key].append(val)


	def new_batch(self, batch): # auto accumulate
		super().new_batch(batch)
		if self._auto_cumulative_keys is not None:
			for key in self._auto_cumulative_keys:
				if key in self:
					self.accumulate(key, self[key])


	def clear_cumulatives(self):
		self._cumulatives.clear()


	class MissingCumulative(KeyError):
		pass


	def aggregate(self, key):
		if key not in self._cumulatives:
			raise self.MissingCumulative(key)
		return torch.cat(self._cumulatives[key])




class SimpleEvaluationTask(BatchedTask):
	evaluation_key = 'scores'
	score_key = 'mean'
	
	
	class ResultsContainer(Cumulative, BatchedTask.ResultsContainer):
		def __init__(self, evaluation_key=None, auto_cumulative_keys=unspecified_argument, **kwargs):
			if auto_cumulative_keys is unspecified_argument and evaluation_key is not None:
				auto_cumulative_keys = [evaluation_key]
			super().__init__(auto_cumulative_keys=auto_cumulative_keys, **kwargs)
			self.evaluation_key = evaluation_key


		def accumulate(self, key=None, val=None):
			if key is None or val is None:
				if val is None:
					val = key
				key = self.evaluation_key
				return super().accumulate(key, val)
			return super().accumulate(key, val)


		def aggregate(self, key=None):
			if key is None:
				key = self.evaluation_key
			return super().aggregate(key)


	@agnosticmethod
	def create_results_container(self, evaluation_key=unspecified_argument, **kwargs):
		if evaluation_key is unspecified_argument:
			evaluation_key = self.evaluation_key
		return super().create_results_container(evaluation_key=evaluation_key, **kwargs)


	@agnosticmethod
	def score_names(self):
		return {'mean', 'std', 'min', 'max', *super().score_names()}


	@agnosticmethod
	def heavy_results(self):
		return {self.evaluation_key, *super().heavy_results()}
	

	@staticmethod
	def aggregate(info):
		info = super().aggregate(info)

		scores = info.aggregate()

		info.update({
			'scores': scores,
			'mean': scores.mean().item(),
			'max': scores.max().item(),
			'min': scores.min().item(),
			'std': scores.std().item(),
		})
		return info






