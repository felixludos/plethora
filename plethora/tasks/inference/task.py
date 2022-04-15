from omnibelt import get_printer, unspecified_argument, agnosticmethod

from ...framework.util import spaces
from ...framework import with_hparams, with_modules, models, with_args
from ..base import Task, BatchedTask, SimpleEvaluationTask
from ...datasets.base import EncodableDataset, BufferView, SupervisedDataset

from .estimators import GradientBoostingBuilder

prt = get_printer(__file__)



@with_modules(encoder=models.Encoder)
class DownstreamTask(Task):

	ObservationBuffer = EncodableDataset.EncodedBuffer
	def _wrap_dataset(self, dataset, encoder=None):
		if encoder is None:
			encoder = self.encoder
		view = dataset.create_view()
		view.register_buffer('observation', self.ObservationBuffer(encoder=encoder,
		                                                     source=dataset.get_buffer('observation')))
		return view


	def create_results_container(self, encoder=unspecified_argument, source=None, dataset=None, **kwargs):
		if encoder is unspecified_argument:
			encoder = self.encoder
		if dataset is not None:
			dataset = self._wrap_dataset(dataset, encoder)
		if source is not None:
			source = self._wrap_dataset(source, encoder)
		return super().create_results_container(encoder=encoder, dataset=dataset, source=source, **kwargs)


	class ResultsContainer(Task.ResultsContainer):
		@property
		def encoder(self):
			return self._encoder
		@encoder.setter
		def encoder(self, encoder):
			self._encoder = encoder
			if self.source is not None:
				self.source.get_buffer('observation').encoder = encoder
			if self.dataset is not None:
				self.dataset.get_buffer('observation').encoder = encoder



@with_args(eval_split=0.2)
@with_modules(builder=models.ModelBuilder, required=False)
class InferenceTask(DownstreamTask):
	class ResultsContainer(DownstreamTask.ResultsContainer):
		@property
		def din(self):
			return self.source.observation_space
		@property
		def dout(self):
			return self.source.target_space


	EstimatorBuilder = GradientBoostingBuilder # default builder using gradient boosting trees
	@agnosticmethod
	def create_results_container(self, builder=None, **kwargs):
		if builder is None:
			builder = self.EstimatorBuilder()
		info = super().create_results_container(builder=builder, **kwargs)
		self._prepare_source(info)
		self._prepare_estimator(info)
		return info
	
	
	@agnosticmethod
	def run(self, info):
		self._train(info)
		self._eval(info)
		return info


	@staticmethod
	def _prepare_source(info, *, shuffle=True):
		info.train_dataset, info.eval_dataset = info.source.split([None, info.eval_split],
		                                                          shuffle=shuffle, register_modes=False)
		return info


	@staticmethod
	def _prepare_estimator(info, *, builder=None, source=None, **kwargs):
		if source is None:
			source = info.source
		if builder is None:
			builder = info.builder
		info.estimator = builder(source=source, **kwargs)
		return info


	@staticmethod
	def _train(info):
		info.merge_results(info.estimator.fit(info.train_dataset))
		return info


	@staticmethod
	def _eval(info):
		info.merge_results(info.estimator.evaluate(info.eval_dataset))
		return info






