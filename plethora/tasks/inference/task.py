from omnibelt import get_printer, unspecified_argument

from ...framework.util import spaces
from ..base import Task, BatchedTask, SimpleEvaluationTask
from ...datasets.base import EncodableDataset, BufferView, SupervisedDataset

prt = get_printer(__file__)



class DownstreamTask(Task):
	def __init__(self, dataset=None, encoder=None, **kwargs):
		self.encoder = encoder
		super().__init__(**kwargs)
		if dataset is not None:
			dataset = self._wrap_dataset(dataset)
		self.dataset = dataset


	ObservationBuffer = EncodableDataset.EncodedBuffer
	def _wrap_dataset(self, dataset):
		dataset = dataset.copy()
		dataset.register_buffer('observation', self.ObservationBuffer(encoder=getattr(self, 'encoder', None),
		                                                     source=dataset.get_buffer('observation')))
		return dataset


	class ResultsContainer(Task.ResultsContainer):
		@property
		def encoder(self):
			return self._encoder
		@encoder.setter
		def encoder(self, encoder):
			self._encoder = encoder
			self.dataset.get_buffer('observation').encoder = encoder



class AbstractInferenceTask(DownstreamTask):
	def __init__(self, encoder=None, eval_split=0.2, **kwargs):
		super().__init__(**kwargs)
		self.encoder = encoder
		self.eval_split = eval_split
	
	
	class ResultsContainer(DownstreamTask.ResultsContainer):
		def train(self):
			self.dataset = self.train_dataset
	
	
		def eval(self):
			self.dataset = self.eval_dataset


		# @Node.importance.setter
		# def importance(self, new_importance):
		# 	# You can change the order of these two lines:
		# 	assert new_importance >= 3
		# 	Node.importance.fset(self, new_importance)


		@property
		def din(self):
			return self.dataset.observation_space


		@property
		def dout(self):
			return self.dataset.target_space
	
	
	def prepare(self, **kwargs):
		info = super().prepare(**kwargs)
		info.encoder = self.encoder
		info.train_dataset, info.eval_dataset = self._prepare_datasets(self.dataset, self.encoder)
		return info
	
	
	@classmethod
	def run(cls, info):
		# info.train()
		cls._train(info)
		# info.eval()
		cls._eval(info)
		return info


	@staticmethod
	def _prepare_datasets(dataset, encoder):
		raise NotImplementedError
	
	
	@staticmethod
	def _train(info):
		raise NotImplementedError


	@staticmethod
	def _eval(info):
		raise NotImplementedError



class InferenceTask(AbstractInferenceTask):
	@staticmethod
	def _train(info):
		info.estimator.fit(info.train_dataset)
		return info


	@staticmethod
	def _eval(info):
		info.estimator.evaluate(info.eval_dataset)
		return info



class Scikit_InferenceTask(AbstractInferenceTask):
	def prepare(self, **kwargs):
		info = super().prepare(**kwargs)
		info.estimator = self.build(info.get_din(), info.get_dout())
		info.use_joint = isinstance(info.estimator, list)
		return info
	

	#
	#
	# class TargetBuffer(WrappedBuffer):
	# 	pass
	#
	#
	# @classmethod
	# def wrap_dataset(cls, dataset, encoder=None):
	# 	obs_buffer = dataset.get_buffer('observation')
	# 	target_buffer = dataset.get_buffer('target')
	#
	# 	dataset = SupervisedDataset().load()
	# 	dataset.register_buffer('observation', cls.ObservationBuffer(encoder=encoder, source=obs_buffer))
	# 	dataset.register_buffer('target', cls.TargetBuffer(source=target_buffer))
	# 	return dataset
	
	
	def _prepare_datasets(self, dataset, encoder):
		train_dataset, eval_dataset = dataset.split([None, self.eval_split], shuffle=True, register_modes=False)
		# train_dataset = self.wrap_dataset(train_dataset, encoder=self.encoder)
		# eval_dataset = self.wrap_dataset(eval_dataset, encoder=self.encoder)
		
		train_dataset.register_buffer('observation',
		                              self.ObservationBuffer(encoder=encoder,
		                                                     source=train_dataset.get_buffer('observation')))
		eval_dataset.register_buffer('observation', self.ObservationBuffer(encoder=encoder,
		                                                     source=eval_dataset.get_buffer('observation')))
		return train_dataset, eval_dataset
	
	
	class MissingEstimatorError(Exception):
		def __init__(self, din, dout):
			super().__init__(f'input: {din}, output: {dout}')
	
	
	@classmethod
	def build(cls, din, dout):
		if isinstance(dout, spaces.JointSpace):
			return [cls.build(din, dim) for dim in dout]
		
		elif isinstance(dout, spaces.CategoricalDim):
			return
		
		
		
		raise cls.MissingEstimatorError(din, dout)



# TODO: joint estimator






