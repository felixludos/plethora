import torch
from omnibelt import get_printer, agnosticmethod, unspecified_argument
from ...framework import abstract, hparam, inherit_hparams
from ...datasets.buffers import BufferView, ReplacementBuffer
from ...datasets.base import LabeledDataset
from ..base import Task, GeneralizationTask
# from .criteria import get_criterion


@inherit_hparams('task', 'baseline_task', 'trial_task')
class AbstractTransferTask(GeneralizationTask, abstract.Augmentation):
	original_key = 'observation'
	augmented_key = 'augmented'

	augmentation = hparam(module=abstract.Augmentation)


	def augment(self, observation):
		return self.augmentation.augment(observation)


	@agnosticmethod
	def unmodify_source(self, source):
		return source.source


	@agnosticmethod
	def modify_source(self, source, augmentation=unspecified_argument, **kwargs):
		if augmentation is unspecified_argument:
			augmentation = self
		source = super().modify_source(source)
		if source is None:
			return source
		view = source.create_view()
		replacement = self.AugmentedBuffer(augmentation=augmentation, key=self.original_key, source=source, **kwargs)
		view.register_buffer(self.augmented_key, replacement)
		# if self.original_key is not None:
		# 	view.register_buffer(self.original_key, self.OriginalBuffer(key=self.original_key, source=source))
		return view


	# OriginalBuffer = ReplacementBuffer

	class AugmentedBuffer(ReplacementBuffer):
		def __init__(self, augmentation=None, **kwargs):
			super().__init__(**kwargs)
			self.augmentation = augmentation


		def _get(self, sel=None, **kwargs):
			samples = super()._get(sel=sel, **kwargs)
			if self.augmentation is not None:
				return self.augmentation.augment(samples)
			return samples



@inherit_hparams('augmentation', 'task', 'baseline_task', 'trial_task')
class EquivarianceTransferTask(AbstractTransferTask): # replace "observation" when loaded
	def modify_task(self, task):
		new = super().modify_task(task)
		new.observation_key = self.augmented_key
		return new



@inherit_hparams('augmentation', 'task', 'baseline_task', 'trial_task')
class AbstractInvarianceTransferTask(AbstractTransferTask): # modify individual operations (eg. encode)
	_modifications = []

	def __init__(self, modifications=None, **kwargs):
		if modifications is None:
			modifications = self._modifications
		super().__init__(**kwargs)
		self._modifications = set(modifications)


	@agnosticmethod
	def modify_task(self, task):
		new = super().modify_task(task)
		for key in self._modifications:
			setattr(new, key, self)
		return new



@inherit_hparams('augmentation', 'task', 'baseline_task', 'trial_task')
class InvariantEncoderTransferTask(AbstractInvarianceTransferTask, abstract.Encoder):
	_modifications = ['encoder']


	@hparam(module=abstract.Encoder)
	def encoder(self):
		return self.trial_task.encoder


	def encode(self, observation):
		augmented = self.trial_task.info[self.augmented_key]
		return self.encoder.encode(augmented)



class AnalogueTransferTask(AbstractTransferTask): # for (synthetic) datasets that are analoguous (eg. MPI3D)
	label_key = 'label'

	analogue_dataset = hparam(module=LabeledDataset) # should not have a train/test split


	@agnosticmethod
	def modify_source(self, source, analogue=unspecified_argument, label_key=unspecified_argument, **kwargs):
		if analogue is analogue:
			analogue = self.analogue_dataset
		if label_key is unspecified_argument:
			label_key = self.label_key
		return super().modify_source(source, analogue=analogue, label_key=label_key, **kwargs)


	class AugmentedBuffer(ReplacementBuffer):
		def __init__(self, analogue=None, label_key=None, **kwargs):
			super().__init__(**kwargs)
			self.analogue = analogue
			self.label_key = label_key


		def _get(self, sel=None, **kwargs):
			# if self.analogue_source is not None:
			label = self.source.get(self.label_key, sel=sel, **kwargs)
			return self.analogue.generate_observation_from_label(label)



@inherit_hparams('analogue_dataset', 'task', 'baseline_task', 'trial_task')
class AnalogueEquiTransferTask(AnalogueTransferTask, EquivarianceTransferTask):
	pass



@inherit_hparams('encoder', 'analogue_dataset', 'task', 'baseline_task', 'trial_task')
class AnalogueInvTransferTask(AnalogueTransferTask, InvariantEncoderTransferTask):
	pass




