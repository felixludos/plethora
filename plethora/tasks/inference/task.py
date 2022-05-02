from omnibelt import get_printer, agnosticmethod

from ...framework import abstract, hparam, inherit_hparams
from ..base import Task
from ...datasets.base import EncodableDataset, DataSource

from .estimators import GradientBoostingWrapperBuilder

prt = get_printer(__file__)



class DownstreamTask(Task):

	encoder = hparam(default=None, module=abstract.Extractor)


	@hparam(cache=True, module=DataSource)
	def dataset(self):
		return self._wrap_dataset(self._dataset, self.encoder)
	@dataset.setter # TODO: fix linting issue
	def dataset(self, dataset):
		self._dataset = dataset


	ObservationBuffer = EncodableDataset.EncodedBuffer
	def _wrap_dataset(self, dataset, encoder=None):
		if encoder is None:
			encoder = self.encoder
		view = dataset.create_view()
		view.register_buffer('observation', self.ObservationBuffer(encoder=encoder, max_batch_size=dataset.batch_size,
		                                                     source=dataset.get_buffer('observation'),))
		return view


	@agnosticmethod
	def create_results_container(self, source=None, **kwargs):
		info = super().create_results_container(source=source, **kwargs)
		if source is not None:
			info.new_source(self._wrap_dataset(source, self.encoder))
		return info



@inherit_hparams('dataset')
class InferenceTask(DownstreamTask):
	train_score_key = 'train_score'

	num_samples = None
	shuffle_split = True
	eval_split = hparam(0.2)

	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self._train_dataset = [] # stack
		self._eval_dataset = [] # stack


	ObservationBuffer = EncodableDataset.EncodedBuffer
	def _wrap_dataset(self, dataset, encoder=None):
		if self.num_samples is not None:
			dataset = dataset.subset(self.num_samples, shuffle=True)
		return super()._wrap_dataset(dataset, encoder=encoder)


	EstimatorBuilder = GradientBoostingWrapperBuilder # default builder using gradient boosting trees
	@hparam(cache=True)
	def builder(self):
		return self.EstimatorBuilder()


	@hparam(cache=True, module=abstract.Estimator)
	def estimator(self):
		return self.builder(source=self.train_dataset)


	@hparam()#module=DataSource)
	def train_dataset(self):
		if len(self._train_dataset) == 0:
			self._split_dataset()
		return self._train_dataset[-1]
	@train_dataset.setter
	def train_dataset(self, dataset):
		self._train_dataset.append(self._wrap_dataset(dataset, self.encoder))
	@train_dataset.deleter
	def train_dataset(self):
		self._train_dataset.clear()


	@hparam()#module=DataSource)
	def eval_dataset(self):
		if len(self._eval_dataset) == 0:
			self._split_dataset()
		return self._eval_dataset[-1]
	@eval_dataset.setter
	def eval_dataset(self, dataset):
		self._eval_dataset.append(self._wrap_dataset(dataset, self.encoder))
	@eval_dataset.deleter
	def eval_dataset(self):
		self._eval_dataset.clear()


	@agnosticmethod
	def _split_dataset(self, source=None): # TODO maybe treat these as hparams
		if source is None:
			source = self.dataset
		train_dataset, eval_dataset = source.split([None, self.eval_split], shuffle=self.shuffle_split)
		self._train_dataset.append(train_dataset)
		self._eval_dataset.append(eval_dataset)


	# @agnosticmethod
	# def create_results_container(self, **kwargs):
	# 	info = super().create_results_container(**kwargs)
	# 	if info.source is not None:
	# 		info.train_dataset, info.eval_dataset = self._split_dataset(info.source)
	# 	return info


	@agnosticmethod
	def _compute(self, info):
		if info.source is not None:
			self._split_dataset(info.source)

		self._train(info)
		self._eval(info)

		if info.source is not None:
			self._train_dataset.pop()
			self._eval_dataset.pop()
		return info


	@agnosticmethod
	def fit(self, dataset):
		return self.estimator.fit(dataset)


	@agnosticmethod
	def _train(self, info):
		out = self.fit(self.train_dataset)
		info.merge_results(out)
		if 'score' in out:
			info[self.train_score_key] = out['score']
		return info


	@agnosticmethod
	def evaluate(self, dataset):
		return self.estimator.evaluate(dataset)


	@agnosticmethod
	def _eval(self, info):
		out = self.evaluate(self.eval_dataset)
		info.merge_results(out)
		info[self.score_key] = out['score']
		return info






