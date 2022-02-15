
from omnibelt import unspecified_argument, duplicate_instance

from .features import Device, DeviceContainer, Loadable, Seeded



class BufferTransform:
	def transform(self, sample=None):
		return sample


# class BufferUpdateFailedError(NotImplementedError):
# 	pass

class NotLoadedError(Exception):
	def __init__(self, buffer):
		super().__init__(f'{buffer}')



class Buffer(BufferTransform, Loadable, DeviceContainer, Seeded):
	space = None
	
	def __init__(self, space=unspecified_argument, transforms=None, default_len=None, **kwargs):
		super().__init__(**kwargs)
		if space is unspecified_argument:
			space = self.space
		if transforms in None:
			transforms = []
		self.space = space
		self.transforms = transforms
		self._waiting_update = None
		self._default_len = default_len


	def copy(self):
		return duplicate_instance(self) # shallow copy


	def set_space(self, space):
		self.space = space


	def get_space(self):
		return self.space


	def _update(self, **kwargs):
		raise NotImplementedError


	def update(self, **kwargs):
		if self.is_loaded():
			return self._update(**kwargs)
		self._waiting_update = self._store_update(**kwargs)


	def load(self, *args, **kwargs):
		out = super().load(*args, **kwargs)
		if self._waiting_update is not None:
			self._apply_update(self._waiting_update)
			self._waiting_update = None
		return out


	def _store_update(self, **kwargs):
		return kwargs


	def _apply_update(self, update):
		return self.update(**update)


	def transform(self, sample=None):
		for transform in self.transforms:
			sample = transform(sample)
		return sample


	def _get(self, sel=None, device=None, **kwargs):
		raise NotImplementedError

	
	def get(self, sel=None, device=None, **kwargs):
		if not self.is_loaded():
			raise NotLoadedError(self)
		sample = self._get(sel, device=device, **kwargs)
		return self.transform(sample)



class FixedBuffer(Buffer): # fixed number of samples (possibly not known until after loading)

	def _get(self, indices=None, device=None, **kwargs):
		return super()._get(indices, device=device, **kwargs)


	def get(self, indices=None, device=None, **kwargs):
		return super().get(indices, device=device, **kwargs)


	def _update(self, indices=None, **kwargs):
		raise NotImplementedError


	def _store_update(self, indices=None, **kwargs):
		if self._waiting_update is not None and indices is not None:
			indices = self._waiting_update[indices]
		return indices


	def _apply_update(self, indices, **kwargs):
		return super()._apply_update(dict(indices=indices, **kwargs))


	def _load_indices(self, indices=None, **kwargs):
		raise NotImplementedError


	def _load(self, **kwargs):
		return self._load_indices(**kwargs)


	def load(self, **kwargs):
		if not self.is_loaded() and self._waiting_update is not None:
			try:
				self._load_indices(indices=self._waiting_update, **kwargs)
			except NotImplementedError:
				self._load(**kwargs)
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
		raise NotLoadedError(self)


	def __len__(self):
		return self.count()


	def __getitem__(self, item):
		return self.get(item)



class UnlimitedBuffer(Buffer):
	pass



class Function(Device):
	din, dout = None, None
	
	def __init__(self, *args, din=unspecified_argument, dout=unspecified_argument, **kwargs):
		super().__init__(*args, **kwargs)
		if din is unspecified_argument:
			din = self.din
		if dout is unspecified_argument:
			dout = self.dout
		self.din, self.dout = din, dout

	
	def get_dims(self):
		return self.din, self.dout
	

class TensorContainer(Device):

	def _item_iterator(self):
		raise NotImplementedError


	def _update_payload(self, updates):
		for key, content in updates.items():
			self[key] = content


	def _to(self, device, **kwargs):
		updates = {key: content.to(device) for key, content in self._item_iterator()}
		self._update_payload(updates)


	def __str__(self):
		return f'{self.__class__.__name__}({self._str_info()})'



class TensorList(TensorContainer, list):
	def _str_info(self):
		num = len(self)
		msg = 'item' if num == 1 else 'items'
		return f'{num} {msg}'


	def _item_iterator(self):
		return enumerate(self)



class TensorDict(TensorContainer, dict):
	def _str_info(self):
		msg = ', '.join(self.keys())
		return msg


	def _item_iterator(self):
		return self.items()









