from omnibelt import get_printer, unspecified_argument, agnosticmethod

from ...framework.util import spaces
from ...framework import models, hparam, inherit_hparams
from ..base import Task, BatchedTask, SimpleEvaluationTask
from ...datasets.base import EncodableDataset, BufferView, SupervisedDataset

from .estimators import GradientBoostingBuilder

prt = get_printer(__file__)



class DownstreamTask(Task):
	
	encoder = hparam(default=None, module=models.Extractor)


	ObservationBuffer = EncodableDataset.EncodedBuffer
	def _wrap_dataset(self, dataset, encoder=None):
		if encoder is None:
			encoder = self.encoder
		view = dataset.create_view()
		view.register_buffer('observation', self.ObservationBuffer(encoder=encoder,
		                                                     source=dataset.get_buffer('observation')))
		return view


	@hparam(cache=True)
	def dataset(self):
		return self._wrap_dataset(self._dataset, self.encoder)
		
	
	@dataset.setter
	def dataset(self, dataset):
		self._dataset = dataset


	# class ResultsContainer(Task.ResultsContainer):
	# 	@property
	# 	def encoder(self):
	# 		return self._encoder
	# 	@encoder.setter
	# 	def encoder(self, encoder):
	# 		self._encoder = encoder
	# 		if self.source is not None:
	# 			self.source.get_buffer('observation').encoder = encoder
	# 		if self.dataset is not None:
	# 			self.dataset.get_buffer('observation').encoder = encoder


@inherit_hparams('dataset')
class InferenceTask(DownstreamTask):
	eval_split = hparam(0.2)
	

	EstimatorBuilder = GradientBoostingBuilder # default builder using gradient boosting trees
	@hparam(cache=True)
	def builder(self):
		return self.EstimatorBuilder()


	@hparam(cache=True)
	def estimator(self):
		return self.builder(source=self.dataset)


	@agnosticmethod
	def create_results_container(self, **kwargs):
		info = super().create_results_container(**kwargs)
		self._prepare_source(info)
		return info
	
	
	@agnosticmethod
	def run(self, info):
		self._train(info)
		self._eval(info)
		return info


	@agnosticmethod
	def _prepare(self, shuffle=True):
		self.train_dataset, self.eval_dataset = self.dataset.split([None, self.eval_split], shuffle=shuffle,
		                                                           register_modes=False)


	@agnosticmethod
	def _train(self, info):
		info.merge_results(self.estimator.fit(self.train_dataset))
		return info


	@agnosticmethod
	def _eval(self, info):
		info.merge_results(self.estimator.evaluate(self.eval_dataset))
		return info






