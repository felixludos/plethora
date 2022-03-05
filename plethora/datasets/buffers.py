
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


	def _load_sel(self, sel=None, **kwargs):
		pass


	def _get(self, sel=None, device=None, **kwargs):
		sample = self.data if sel is None else self.data[sel]
		if device is not None:
			sample = sample.to(device)
		return sample


	def _update(self, sel=None, **kwargs):
		if sel is not None:
			self.data = self.data[sel]



class HDFBuffer(base.FixedBuffer):
	def __init__(self, dataset_name=None, path=None, default_len=None, shape=None, **kwargs):
		if path is not None and path.exists() and dataset_name is not None:
			with hf.File(str(path), 'r') as f:
				shape = f[dataset_name].shape
		if default_len is None:
			default_len = shape[0]
		super().__init__(default_len=default_len, **kwargs)
		self.path = path
		self.key_name = dataset_name

		self._shape = shape
		self._selected = None
		self.register_children(_selected=None)


	def _count(self):
		return self._shape[0] if self._selected is None else len(self._selected)


	def _load_sel(self, sel=None, **kwargs):
		pass


	def _update(self, sel=None, **kwargs):
		if sel is not None:
			self._selected = sel if self._selected is None else self._selected[sel]


	def _get(self, sel=None, device=None, **kwargs):
		if sel is None:
			sel = ()
		else:
			if self._selected is not None:
				sel = self._selected[sel]
			sel = torch.as_tensor(sel).numpy()

		with hf.File(str(self.path), 'r') as f:
			sample = f[self.key_name][sel]
		sample = torch.as_tensor(sample)
		if device is not None:
			sample = sample.to(device)
		return sample



class LoadableHDFBuffer(TensorBuffer, HDFBuffer):
	def _load_sel(self, sel=None, **kwargs):
		data = super(TensorBuffer, self)._get(sel=sel, **kwargs)
		self.set_data(data)



class WrappedBuffer(TensorBuffer):
	def __init__(self, source=None, sel=None, **kwargs):
		super().__init__(**kwargs)
		self.source, self.sel = None, None
		self.register_children(source=source, sel=sel)
		self.set_source(source)


	def is_loaded(self):
		return (self.source is not None and self.source.is_loaded()) or (self.source is None and super().is_loaded())


	def _count(self):
		if self.sel is None:
			return (self.source is not None and len(self.source)) or (self.source is None and super()._count())
		return len(self.sel)


	def unwrap(self, **kwargs):
		if self.is_loaded() and self.source is not None:
			self.set_data(self._get(sel=self.sel, device=self.device, **kwargs))
			self.sel = None
			self.set_space(self.get_space())
			# self._loaded = self.source._loaded
			# self.set_source()
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


	def _update(self, sel=None, **kwargs):
		if self.source is None:
			super()._update(sel=sel, **kwargs)


	def _get(self, sel=None, device=None, **kwargs):
		if self.data is not None:
			return super()._get(sel, device=device, **kwargs)
		if self.sel is not None:
			sel = self.sel if sel is None else self.sel[sel]
		assert self.source is not None, 'missing source'
		return self.source.get(sel, device=device, **kwargs)






