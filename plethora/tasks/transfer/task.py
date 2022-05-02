import torch
from omnibelt import get_printer, agnosticmethod, unspecified_argument
from ...framework import abstract, hparam, inherit_hparams
from ...datasets.buffers import BufferView, ReplacementBuffer
from ...datasets.base import LabeledDataset
from ..base import Task, GeneralizationTask
# from .criteria import get_criterion


@inherit_hparams('task', 'baseline_task', 'trial_task')
class AbstractTransferTask(abstract.Augmentation, GeneralizationTask):

	augmentation = hparam(module=abstract.Augmentation)


	def augment(self, observation):
		return self.augmentation.augment(observation)



@inherit_hparams('augmentation', 'task', 'baseline_task', 'trial_task')
class EquivarianceTransferTask(AbstractTransferTask): # replace "observation" when loaded
	buffer_key = 'observation'


	@agnosticmethod
	def modify_source(self, source, augmentation=unspecified_argument, **kwargs):
		if augmentation is unspecified_argument:
			augmentation = self
		source = super().modify_source(source)
		if source is None:
			return source
		view = source.create_view()
		replacement = self.AugmentedBuffer(augmentation=augmentation, key=self.buffer_key, source=source, **kwargs)
		view.register_buffer(self.buffer_key, replacement)
		return view


	@agnosticmethod
	def unmodify_source(self, source):
		return source.source


	class AugmentedBuffer(ReplacementBuffer):
		def __init__(self, augmentation=None, **kwargs):
			super().__init__(**kwargs)
			self.augmentation = augmentation


		def _get(self, sel=None, **kwargs):
			samples = super()._get(sel=sel, **kwargs)
			if self.augmentation is not None:
				return self.augmentation.augment(samples)
			return samples



class AnalogueEquiTransferTask(EquivarianceTransferTask): # for (synthetic) datasets that are analoguous (eg. MPI3D)
	label_key = 'label'

	analogue_dataset = hparam(module=LabeledDataset)


	@agnosticmethod
	def modify_source(self, source, analogue_source=unspecified_argument, label_key=unspecified_argument, **kwargs):
		if analogue_source is unspecified_argument:
			analogue_source = self.analogue_source
		if label_key is unspecified_argument:
			label_key = self.label_key
		return super().modify_source(source, analogue_source=analogue_source, label_key=label_key, **kwargs)


	class AugmentedBuffer(ReplacementBuffer):
		def __init__(self, analogue_source=None, label_key=None, **kwargs):
			super().__init__(**kwargs)
			self.analogue_source = analogue_source
			self.label_key = label_key


		def _get(self, sel=None, **kwargs):
			# if self.analogue_source is not None:
			label = self.source_table.get(self.label_key, sel=sel, **kwargs)
			return self.analogue_source.generate_observation_from_label(label)



@inherit_hparams('augmentation', 'task', 'baseline_task', 'trial_task')
class InvarianceTransferTask(AbstractTransferTask): # modify operations manually (eg. encode)
	def __init__(self, **kwargs):

		pass


	@agnosticmethod
	def modify_task(self, task):
		new = super().modify_task(task)


	pass



class EncoderTransferTask(InvarianceTransferTask):

	pass








