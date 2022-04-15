
import torch

from omnibelt import get_printer, agnosticmethod, unspecified_argument

from ..framework.features import Seeded, Prepared, with_args
from ..framework.base import Container
from ..framework.models import Computable

prt = get_printer(__file__)


@with_args(slim=False, online=False, dataset=None)
class Task(Computable, Prepared, Seeded): # TODO: specify random seed for reproducibility
	@agnosticmethod
	def score_names(self):
		return set() if self.score_key is None else {'score'}


	@agnosticmethod
	def heavy_results(self):
		return set()

	
	class ResultsContainer(Computable.ResultsContainer):
		@property
		def source(self):
			return self._source if self._source is None else self.dataset
		@source.setter
		def source(self, source):
			self._source = source


	def _prepare(self, *args, **kwargs):
		pass


	@agnosticmethod
	def create_results_container(self, seed=unspecified_argument, **kwargs):
		if seed is unspecified_argument:
			seed = self._seed
		return super().create_results_container(seed=seed, **kwargs)
		
		
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
		# self.register_arg('num_samples', 'batch_size', 'force_batch_size', 'hard_sample_limit', 'pbar')


	# class ResultsContainer(Task.ResultsContainer):
	# 	@property
	# 	def source(self): # prevent source (batch) from defaulting to full dataset
	# 		return self._source
	# 	@source.setter
	# 	def source(self, source):
	# 		self._source = source


	@agnosticmethod
	def loop(self, info):
		if info.source is not None:
			info.new_source(info.source)
			yield info
			return

		itr = info.dataset.get_iterator(num_samples=self.num_samples, batch_size=self.batch_size, pbar=self.pbar,
		                                force_batch_size=self.force_batch_size, hard_limit=self.hard_sample_limit,
		                                gen=info.gen)
		for batch in itr:
			info.new_source(batch)
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
	_auto_cumulative_keys = set()

	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self._auto_cumulative_keys = self._auto_cumulative_keys.copy()
		self._cumulatives = {}


	@agnosticmethod
	def register_cumulative(self, *keys):
		self._auto_cumulative_keys.update(keys)


	def accumulate(self, key, val):
		if key not in self._cumulatives:
			self._cumulatives[key] = []
		self._cumulatives[key].append(val)


	def auto_accumulate(self):
		for key in self._auto_cumulative_keys:
			if key in self:
				self.accumulate(key, self[key])


	def new_source(self, source): # auto accumulate
		super().new_source(source)
		self.auto_accumulate()


	def clear_cumulatives(self):
		self._cumulatives.clear()


	class MissingCumulative(KeyError):
		pass


	def aggregate(self, key):
		if key not in self._cumulatives:
			raise self.MissingCumulative(key)
		return torch.cat(self._cumulatives[key])



@with_args(evaluation_key='scores')
class SimpleEvaluationTask(BatchedTask):
	score_key = 'mean'

	class ResultsContainer(Cumulative, BatchedTask.ResultsContainer):
		def __init__(self, evaluation_key=None, **kwargs):
			super().__init__(**kwargs)
			if evaluation_key is not None:
				self.register_cumulative(evaluation_key)
			self.evaluation_key = evaluation_key


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
		heavy = super().heavy_results()
		if self.evaluation_key is not None:
			heavy.add(self.evaluation_key)
		return heavy
	

	@staticmethod
	def aggregate(info):
		info = super().aggregate(info)

		scores = info.aggregate(info.evaluation_key)

		info.update({
			info.evaluation_key: scores,
			'mean': scores.mean().item(),
			'max': scores.max().item(),
			'min': scores.min().item(),
			'std': scores.std().item(),
		})
		return info






