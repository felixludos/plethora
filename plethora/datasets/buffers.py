
import math
import torch
import h5py as hf
from ..framework import base, Fileable



class TensorBuffer(base.FixedBuffer):
	def __init__(self, data=None, **kwargs):
		super().__init__(**kwargs)
		self.data = None
		self.set_data(data)


	def set_data(self, data=None):
		# self.device = None if data is None else data.device
		self.register_children(data=data)


	def is_loaded(self):
		return self.data is not None


	def _count(self):
		return len(self.data)


	def _load_indices(self, indices=None, **kwargs):
		pass


	def _get(self, indices=None, device=None, **kwargs):
		sample = self.data if indices is None else self.data[indices]
		if device is not None:
			sample = sample.to(device)
		return sample


	def _update(self, indices=None, **kwargs):
		if indices is not None:
			self.data = self.data[indices]



class HDFBuffer(base.FixedBuffer):
	def __init__(self, dataset_name, path, default_len=None, **kwargs):
		with hf.File(str(path), 'r') as f:
			shape = f[dataset_name].shape
		if default_len is None:
			default_len = shape[0]
		super().__init__(default_len=default_len, **kwargs)
		self.path = path
		self.key_name = dataset_name

		self._shape = shape
		self._sel_indices = None
		self.register_children(_sel_indices=None)


	def _count(self):
		return self._shape[0] if self._sel_indices is None else len(self._sel_indices)


	def _load_indices(self, indices=None, **kwargs):
		pass


	def _update(self, indices=None, **kwargs):
		if indices is not None:
			self._sel_indices = indices if self._sel_indices is None else self._sel_indices[indices]


	def _get(self, indices=None, device=None, **kwargs):
		if indices is None:
			indices = ()
		else:
			if self._sel_indices is not None:
				indices = self._sel_indices[indices]
			indices = torch.as_tensor(indices).numpy()

		with hf.File(str(self.path), 'r') as f:
			sample = f[self.key_name][indices]
		sample = torch.as_tensor(sample)
		if device is not None:
			sample = sample.to(device)
		return sample



class LoadableHDFBuffer(TensorBuffer, HDFBuffer):
	def _load_indices(self, indices=None, **kwargs):
		data = super(TensorBuffer, self)._get(indices=indices, **kwargs)
		self.set_data(data)



class WrappedBuffer(TensorBuffer):
	def __init__(self, source=None, indices=None, space=None, data=None, **kwargs):
		super().__init__(space=None, data=None, **kwargs)
		self.source, self.indices = None, None
		self.register_children(source=source, indices=indices)
		self.set_source(source)


	def is_loaded(self):
		return (self.source is not None and self.source.is_loaded()) or (self.source is None and super().is_loaded())


	def _count(self):
		if self.indices is None:
			return (self.source is not None and len(self.source)) or (self.source is None and super()._count())
		return len(self.indices)


	def unwrap(self, **kwargs):
		if self.is_loaded() and self.source is not None:
			self.set_data(self._get(self.indices, device=self.device, **kwargs))
			self.indices = None
			self.set_space(self.get_space())
			self._loaded = self.source._loaded
			self.set_source()
			return
		raise base.NotLoadedError(self)


	def merge(self, new_instance=None):
		raise NotImplementedError


	@staticmethod
	def stack(*datasets): # TODO: append these
		raise NotImplementedError


	def set_source(self, source=None):
		self.source = source


	def get_space(self):
		if self.space is None:
			return self.source.get_space()
		return self.space


	def _load(self, *args, **kwargs):
		pass


	def _update(self, indices=None, **kwargs):
		if self.source is None:
			super()._update(indices, **kwargs)


	def _get(self, indices=None, device=None, **kwargs):
		if self.source is None:
			return super()._get(indices, device=device, **kwargs)
		if self.indices is not None:
			indices = self.indices if indices is None else self.indices[indices]
		return self.source.get(indices, device=device, **kwargs)






