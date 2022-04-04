
import math
import torch
import h5py as hf
from ..framework import base, Sourced


class FixedBuffer(base.Buffer): # fixed number of samples (possibly not known until after loading)

	# def _get(self, indices=None, device=None, **kwargs):
	# 	return super()._get(indices, device=device, **kwargs)
	#
	#
	# def get(self, indices=None, device=None, **kwargs):
	# 	return super().get(indices, device=device, **kwargs)


	def _update(self, sel=None, **kwargs):
		raise NotImplementedError


	def _store_update(self, sel=None, **kwargs):
		if self._waiting_update is not None and sel is not None:
			sel = self._waiting_update[sel]
		return sel


	def _apply_update(self, sel, **kwargs):
		return super()._apply_update(dict(sel=sel, **kwargs))


	def _load_sel(self, sel=None, **kwargs):
		raise NotImplementedError


	def _load(self, **kwargs):
		return self._load_sel(**kwargs)


	def load(self, **kwargs):
		if not self.is_loaded() and self._waiting_update is not None:
			try:
				self._load_sel(sel=self._waiting_update, **kwargs)
			except NotImplementedError:
				pass # _load + _update will be called in super().load
			else:
				self._waiting_update = None
		return super().load(**kwargs)


	def _count(self):
		raise NotImplementedError


	def count(self):
		if self.is_loaded():
			return self._count()
		if self._waiting_update is not None:
			return len(self._waiting_update)
		if self._default_len is not None:
			return self._default_len
		raise self.NotLoadedError(self)


	def __len__(self):
		return self.count()


	def __getitem__(self, item):
		return self.get(item)



class UnlimitedBuffer(base.Buffer):
	pass



class TensorBuffer(FixedBuffer):
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



class HDFBuffer(FixedBuffer):
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
			self.space = self.space
			# self._loaded = self.source._loaded
			# self.set_source()
			return
		raise self.NotLoadedError(self)


	def merge(self, new_instance=None):
		raise NotImplementedError


	@staticmethod
	def stack(*datasets): # TODO: append these
		raise NotImplementedError


	def set_source(self, source=None):
		self.source = source


	@property
	def space(self):
		if self._space is None:
			return self.source.space
		return self.space
	@space.setter
	def space(self, space):
		self._space = space


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






