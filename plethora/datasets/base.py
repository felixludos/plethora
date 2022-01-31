import torch

from omnibelt import unspecified_argument, duplicate_instance

from ..framework import base



class MissingModeError(Exception):
	pass



class BufferNotFoundError(Exception):
	pass



class TensorBuffer(base.Buffer):
	def __init__(self, data=None, **kwargs):
		super().__init__(**kwargs)
		self.data = None
		self.set_data(data)


	def set_data(self, data=None):
		self.device = None if data is None else data.device
		self.data = data


	def _to(self, device):
		self.data = self.data.to(device)


	def _load(self):
		pass


	def _get(self, idx, *args, **kwargs):
		return self.data[idx]


	def _update(self, idx, *args, **kwargs):
		self.data = self.data[idx]



class ReferencedBuffer(base.Buffer):
	def __init__(self, ref=None, **kwargs):
		super().__init__(**kwargs)
		self.ref = ref


	def _load(self, *args, **kwargs):
		pass


	def _get(self, idx, *args, **kwargs):
		return self._collection.buffers.get(self.ref, {}).get(idx)


	def _update(self, idx, *args, **kwargs):
		pass



class DataCollection(base.Buffer):
	sample_format = None
	
	def __init__(self, *, sample_format=unspecified_argument, batch_device=unspecified_argument, mode=None,
	             buffers=None, modes=None, space=None, **kwargs):
		super().__init__(space=None, **kwargs)
		if sample_format is unspecified_argument:
			sample_format = self.sample_format
		self.set_sample_format(sample_format)

		if batch_device is unspecified_argument:
			batch_device = None
		self.batch_device = batch_device

		if buffers is None:
			buffers = {}
		self.buffers = buffers

		self.mode = mode
		if modes is None:
			modes = {}
			if mode is not None:
				modes[mode] = self
		self._modes = modes


	def register_buffer(self, name, buffer=None, space=unspecified_argument, **kwargs):
		if not isinstance(buffer, base.Buffer):
			buffer = TensorBuffer(data=buffer, **kwargs)
		buffer.set_collection(self)
		if space is not unspecified_argument:
			buffer.set_space(space)
		self.buffers[name] = buffer
		return self.buffers[name]


	def get_space(self, name=None):
		if name not in self.buffers:
			raise BufferNotFoundError(name)
		return self.buffers[name].get_space()


	def _load(self, *args, **kwargs):
		super()._load(*args, **kwargs)

		for name, buffer in self.buffers.items():
			buffer.load()


	def _check_buffer_names(self, *names):
		missing = []
		for name in names:
			if name not in self.buffers:
				missing.append(name)
		if len(missing):
			raise BufferNotFoundError(', '.join(missing))
	

	def _get(self, idx=None, sample_format=None, device=None, strict=True):
		if sample_format is None:
			sample_format = self.sample_format
		if device is None:
			device = self.batch_device

		if isinstance(sample_format, str):
			if strict:
				self._check_buffer_names(sample_format)
			batch = self.buffers[sample_format].get(idx, device=device)
		elif isinstance(sample_format, (list, tuple)):
			if strict:
				self._check_buffer_names(*sample_format)
			batch = [self.buffers[name].get(idx, device=device) for name in sample_format]
		elif isinstance(sample_format, set):
			if strict:
				self._check_buffer_names(*sample_format)
			batch = {name:self.buffers[name].get(idx, device=device)
			         for name in sample_format if name in self.buffers}
		elif isinstance(sample_format, dict):
			if strict:
				self._check_buffer_names(*sample_format)
			batch = {}
			for name, transforms in sample_format.items():
				if name in self.buffers:
					samples = self.buffers[name].get(idx)
					if transforms is not None and not isinstance(transforms, (list, tuple)):
						transforms = [transforms]
					for transform in transforms:
						samples = transform.transform(samples)
					batch[name] = samples
		else:
			raise NotImplementedError(f'bad sample format: {sample_format}')

		return batch


	def set_sample_format(self, sample_format):
		self.sample_format = sample_format


	def get_sample_format(self):
		return self.sample_format


	def batch(self, N=None):
		raise NotImplementedError


	def get_mode(self, mode='train'):
		if self.mode == mode:
			return self
		if mode in self._modes:
			return self._modes[mode]
		raise MissingModeError


	
class Dataset(DataCollection):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self._waiting_subset = None
		self._subset_indices = None

	def __iter__(self):
		self.load()
		return self

	def __next__(self):
		pass

	def __len__(self):
		raise NotImplementedError
	

	def subset(self, N=None, indices=None, shuffle=False):
		pass


	def get_subset(self, N=None, indices=None, shuffle=False):
		if N is not None:
			assert N != 0
			if isinstance(N, float):
				assert 0 < N < 1
				N = int(N * len(self))
			if N < 0:
				indices = torch.arange(len(self))[N:]
			else:
				indices = torch.arange(len(self))[:N]
		assert indices is not None, 'no info'
		
		
		
		raise NotImplementedError
	
	
	def split(self, *ratios, **named_ratios):
		assert len(ratios) == 0 or len(named_ratios) == 0, 'cant mix named and unnamed splits'

		raise NotImplementedError




class LabeledDataset(DataCollection):
	def get_label_space(self, N=None):
		raise NotImplementedError


	def get_labels(self, N=None):
		raise NotImplementedError



class DisentanglementDataset(LabeledDataset):
	def get_mechanism_space(self):
		raise NotImplementedError


	def transform_to_mechanisms(self, data):
		return self.get_mechanism_space().transform(data, self.get_label_space())


	def transform_to_labels(self, data):
		return self.get_label_space().transform(data, self.get_mechanism_space())


	def get_observations_from_labels(self, labels):
		raise NotImplementedError


	def difference(self, a, b, standardize=None):
		if standardize is None:
			standardize = self._standardize_scale
		if not self.uses_mechanisms():
			a, b = self.transform_to_mechanisms(a), self.transform_to_mechanisms(b)
		return self.get_mechanism_space().difference(a,b, standardize=standardize)


	def distance(self, a, b, standardize=None):
		if standardize is None:
			standardize = self._standardize_scale
		if not self.uses_mechanisms():
			a, b = self.transform_to_mechanisms(a), self.transform_to_mechanisms(b)
		return self.get_mechanism_space().distance(a,b, standardize=standardize)





