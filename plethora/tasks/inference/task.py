from omnibelt import get_printer, unspecified_argument, agnosticmethod

from ...framework.util import spaces
from ...framework import models, hparam, inherit_hparams
from ..base import Task, BatchedTask, SimpleEvaluationTask
from ...datasets.base import EncodableDataset, BufferView, SupervisedDataset

from .estimators import GradientBoostingBuilder

prt = get_printer(__file__)



class DownstreamTask(Task):

	encoder = hparam(default=None, module=models.Extractor)


	@hparam(cache=True)
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
		view.register_buffer('observation', self.ObservationBuffer(encoder=encoder,
		                                                     source=dataset.get_buffer('observation')))
		return view


	@agnosticmethod
	def create_results_container(self, source=None, **kwargs):
		info = super().create_results_container(source=source, **kwargs)
		if source is not None:
			info.new_source(self._wrap_dataset(source, self.encoder))



@inherit_hparams('dataset')
class InferenceTask(DownstreamTask):
	train_score_key = 'train_score'

	shuffle_split = True
	eval_split = hparam(0.2)

	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self._train_dataset = None
		self._eval_dataset = None
	

	EstimatorBuilder = GradientBoostingBuilder # default builder using gradient boosting trees
	@hparam(cache=True)
	def builder(self):
		return self.EstimatorBuilder()


	@hparam(cache=True)
	def estimator(self):
		return self.builder(source=self.train_dataset)


	@agnosticmethod
	def create_results_container(self, source=None, **kwargs):
		info = super().create_results_container(source=source, **kwargs)
		if info.source is not None:
			self._split_dataset(info.source)
		return info


	@hparam()
	def train_dataset(self):
		if self._train_dataset is None:
			self._split_dataset()
		return self._train_dataset
	@train_dataset.deleter
	def train_dataset(self):
		self._train_dataset = None


	@hparam()
	def eval_dataset(self):
		if self._eval_dataset is None:
			self._split_dataset()
		return self._eval_dataset
	@eval_dataset.deleter
	def eval_dataset(self):
		self._eval_dataset = None


	@agnosticmethod
	def _split_dataset(self, source=None): # TODO maybe treat these as hparams
		if source is None:
			source = self.dataset
		self._train_dataset, self._eval_dataset = source.split([None, self.eval_split], shuffle=self.shuffle_split,
		                                                           register_modes=False)


	@agnosticmethod
	def create_results_container(self, source=None, **kwargs):
		info = super().create_results_container(source=source, **kwargs)
		if source is not None:
			self._split_dataset(source)
		return info


	@agnosticmethod
	def _compute(self, info):
		self._train(info)
		self._eval(info)
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






