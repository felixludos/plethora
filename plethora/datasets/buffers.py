
import math
import torch
import h5py as hf
from ..framework import base, Rooted, DeviceContainer


class TransformableBuffer(base.AbstractBuffer):
	transforms = []
	def __init__(self, transforms=None, **kwargs):
		super().__init__(**kwargs)
		if transforms is None:
			transforms = self.transforms.copy()
		self.transforms = transforms


	def transform(self, sample=None):
		for transform in self.transforms:
			sample = transform(sample)
		return sample


	def get(self, sel=None, **kwargs):
		return self.transform(super().get(sel=sel, **kwargs))



class AbstractCountableData(base.AbstractData):
	def __init__(self, default_len=None, **kwargs):
		super().__init__(**kwargs)
		self._default_len = default_len


	def __str__(self):
		return f'{super().__str__()}[{self.size()}]'


	def _size(self):
		raise NotImplementedError


	class UnknownCount(base.AbstractBuffer.NotReady):
		def __init__(self):
			super().__init__('did you forget to provide a "default_len" in __init__?')


	def size(self):
		if self.is_ready:
			return self._size()
		if self._default_len is not None:
			return self._default_len
		raise self.UnknownCount()


	def __len__(self):
		return self.size()


class AbstractCountableDataView(AbstractCountableData, base.AbstractView):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self._default_len = None


	def size(self, **kwargs):
		if self.sel is not None:
			return len(self.sel)
		return super().size()


	def _size(self, **kwargs):
		if self.source is None:
			raise self.NoSource
		return self.source.size(**kwargs)



class AbstractFixedBuffer(AbstractCountableData, base.AbstractBuffer): # fixed number of samples (possibly not known until after loading)
	def _update(self, sel=None, **kwargs):
		raise NotImplementedError


	def size(self):
		try:
			return super().size()
		except self.UnknownCountError:
			if self._waiting_update is not None:
				return len(self._waiting_update)
			raise self.UnknownCountError()


	def _store_update(self, sel=None, **kwargs):
		if self._waiting_update is not None and sel is not None:
			sel = self._waiting_update[sel]
		return sel


	def _apply_update(self, sel, **kwargs):
		return super()._apply_update(dict(sel=sel, **kwargs))


	def _prepare_sel(self, sel=None, **kwargs):
		raise NotImplementedError


	def _prepare(self, **kwargs):
		return self._prepare_sel(**kwargs)


	def prepare(self, **kwargs):
		if not self.is_ready and self._waiting_update is not None:
			try:
				self._prepare_sel(sel=self._waiting_update, **kwargs)
			except NotImplementedError:
				pass # _load + _update will be called in super().load
			else:
				self._waiting_update = None
		return super().prepare(**kwargs)


	def __getitem__(self, item):
		return self.get(item)



# class UnlimitedBuffer(base.AbstractBuffer): # TODO: streams
# 	pass



class Buffer(AbstractFixedBuffer, DeviceContainer):
	def __init__(self, data=None, **kwargs):
		super().__init__(**kwargs)
		self._register_deviced_children(_data=None)
		self.data = data


	def _title(self):
		return f'{self.__class__.__name__}'


	@property
	def data(self):
		return self._data
	@data.setter
	def data(self, data):
		self._data = data

	@property
	def is_ready(self):
		return self.data is not None


	def _size(self):
		return len(self.data)


	class MissingData(AbstractFixedBuffer.NotReady):
		pass


	def _prepare_sel(self, sel=None, **kwargs):
		if self.data is None:
			raise self.MissingData


	def _get(self, sel=None, device=None, **kwargs):
		sample = self.data if sel is None else self.data[sel]
		if device is not None:
			sample = sample.to(device)
		return sample


	def _update(self, sel=None, **kwargs):
		if sel is not None:
			self.data = self.data[sel]



class RemoteBuffer(Buffer):
	def __init__(self, auto_load=True, **kwargs):
		super().__init__(**kwargs)
		self._auto_load = auto_load
		# self._auto_cache = auto_cache


	def _prepare_sel(self, sel=None, **kwargs):
		if self._auto_load:
			self.data = self._get(sel=sel)


	def to_memory(self, **kwargs):
		self.data = self.get(**kwargs)


	def size(self):
		if self.data is not None:
			return super(RemoteBuffer, self)._size()
		return super().size()


	def _get(self, sel=None, **kwargs):
		if self.data is None:
			return super(Buffer, self)._get(sel=sel, **kwargs)
		return super()._get(sel=sel, **kwargs)


	def _update(self, sel=None, **kwargs):
		if self.data is None:
			return super(Buffer, self)._update(sel=sel, **kwargs)
		return super()._update(sel=sel, **kwargs)



class HDFBuffer(RemoteBuffer):
	def __init__(self, dataset_name=None, path=None, default_len=None, shape=None, auto_load=True, **kwargs):
		if path is not None and path.exists() and dataset_name is not None:
			with hf.File(str(path), 'r') as f:
				if dataset_name in f:
					shape = f[dataset_name].shape
				else:
					print('WARNING: could not infer shape') # TODO: use logging
		if default_len is None:
			default_len = shape[0]
		super().__init__(default_len=default_len, **kwargs)
		self.path = path
		self.key_name = dataset_name

		self._auto_load = auto_load

		self._shape = shape
		self._selected = None
		self._register_deviced_children(_selected=None)


	class BadPathError(RemoteBuffer.NotReady):
		pass


	class MissingDatasetError(RemoteBuffer.NotReady):
		pass


	def _prepare_sel(self, sel=None, **kwargs):
		if self.data is None:
			if not self.path.exists():
				raise self.BadPathError(str(self.path))
			with hf.File(str(self.path), 'r') as f:
				if self.key_name not in f:
					raise self.MissingDatasetError(self.key_name)
		super()._prepare_sel(sel=sel, **kwargs)


	def _size(self):
		return self._shape[0] if self._selected is None else len(self._selected)


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



# class LoadableHDFBuffer(TensorBuffer, HDFBuffer):
# 	def _prepare_sel(self, sel=None, **kwargs):
# 		data = super(TensorBuffer, self)._get(sel=sel, **kwargs)
# 		self.set_data(data)



class BufferView(AbstractCountableDataView, RemoteBuffer):

	def _prepare_sel(self, sel=None, **kwargs):
		sel = self._merge_sel(sel)
		if self.source is None:
			raise self.NoSource
		return self.source.prepare(sel=sel, **kwargs)


	@property
	def space(self):
		if self._space is None and self.source is not None:
			return self.source.space
		return self._space
	@space.setter
	def space(self, space):
		self._space = space



	# def __init__(self, source=None, sel=None, **kwargs):
	# 	super().__init__(**kwargs)
	# 	self.source, self.sel = None, None
	# 	self._register_deviced_children(source=source, sel=sel)
	# 	self.set_source(source)

	# def is_ready(self):
	# 	return (self.source is not None and self.source.is_ready()) or (self.source is None and super().is_ready())
	#
	#
	# def _count(self):
	# 	if self.sel is None:
	# 		return (self.source is not None and len(self.source)) or (self.source is None and super()._count())
	# 	return len(self.sel)
	#
	#
	# def unwrap(self, **kwargs):
	# 	if self.is_ready() and self.source is not None:
	# 		self.set_data(self._get(sel=self.sel, device=self.device, **kwargs))
	# 		self.sel = None
	# 		self.space = self.space
	# 		# self._loaded = self.source._loaded
	# 		# self.set_source()
	# 		return
	# 	raise self.NotLoadedError(self)
	#
	#
	# def merge(self, new_instance=None):
	# 	raise NotImplementedError
	#
	#
	# @staticmethod
	# def stack(*datasets): # TODO: append these
	# 	raise NotImplementedError
	#
	#
	# def set_source(self, source=None):
	# 	self.source = source


	# @property
	# def space(self):
	# 	if self._space is None:
	# 		return self.source.space
	# 	return self.space
	# @space.setter
	# def space(self, space):
	# 	self._space = space
	#
	#
	# def _prepare(self, *args, **kwargs):
	# 	pass
	#
	#
	# def _update(self, sel=None, **kwargs):
	# 	if self.source is None:
	# 		super()._update(sel=sel, **kwargs)
	#
	#
	# def _get(self, sel=None, device=None, **kwargs):
	# 	if self.data is not None:
	# 		return super()._get(sel, device=device, **kwargs)
	# 	if self.sel is not None:
	# 		sel = self.sel if sel is None else self.sel[sel]
	# 	assert self.source is not None, 'missing source'
	# 	return self.source.get(sel, device=device, **kwargs)


Buffer.View = BufferView


class Narrow(base.AbstractBuffer):
	def __init__(self, start=None, size=1, dim=1, **kwargs):
		super().__init__(**kwargs)
		self._dim = dim
		self._start = start
		self._size = size


	def _get(self, sel=None, **kwargs):
		sample = super()._get(sel=sel, **kwargs)
		if self._start is not None:
			sample = sample.narrow(self._dim, self._start, self._size)
		return sample



class NarrowBuffer(Narrow, Buffer):
	pass



class NarrowBufferView(Narrow, BufferView):
	pass





