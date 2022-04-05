
from collections import OrderedDict
import torch
from omnibelt import unspecified_argument, duplicate_instance

from .features import Device, DeviceContainer, Loadable, Seeded



class BufferTransform:
	def transform(self, sample=None):
		return sample



class AbstractBuffer(BufferTransform, Loadable, DeviceContainer, Seeded):
	def __init__(self, space=None, transforms=None, default_len=None, **kwargs):
		super().__init__(**kwargs)
		if transforms is None:
			transforms = []
		self.space = space
		self.transforms = transforms
		self._waiting_update = None
		self._default_len = default_len


	def copy(self):
		return duplicate_instance(self) # shallow copy


	@property
	def space(self):
		return self._space
	@space.setter
	def space(self, space):
		self._space = space


	def _update(self, **kwargs):
		raise NotImplementedError


	def update(self, **kwargs):
		if self.is_loaded():
			return self._update(**kwargs)
		self._waiting_update = self._store_update(**kwargs)


	def _load(self, *args, **kwargs):
		pass


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


	class NotLoadedError(Exception):
		def __init__(self, buffer):
			super().__init__(f'{buffer}')


	def get(self, sel=None, device=None, **kwargs):
		if not self.is_loaded():
			raise NotLoadedError(self)
		sample = self._get(sel=sel, device=device, **kwargs)
		return self.transform(sample)



class Function(Device):
	din, dout = None, None

	def __init__(self, *args, din=unspecified_argument, dout=unspecified_argument, **kwargs):
		super().__init__(*args, **kwargs)
		if din is not unspecified_argument:
			self.din = din
		if dout is not unspecified_argument:
			self.dout = dout
		self.din, self.dout = din, dout

	
	def get_dims(self):
		return self.din, self.dout
	


class Container(Device, OrderedDict):
	def _find_missing(self, key):
		raise KeyError(key)


	def _to(self, device, **kwargs):
		for key, val in self.items():
			if isinstance(val, (Device, torch.Tensor)):
				self[key] = val.to(device)


	def _package(self, data):
		return data


	def get(self, key, default=None):
		try:
			val = self[key]
		except KeyError:
			return default
		return self._package(val)


	def __getitem__(self, item):
		if item not in self:
			return self._find_missing(item)
		return super().__getitem__(item)


	def export(self):
		raise NotImplementedError


	def __str__(self):
		entries = ', '.join(self.keys())
		return f'{self.__class__.__name__}({entries})'


	def __repr__(self):
		return str(self)



# class TensorContainer(Device):
#
# 	def _item_iterator(self):
# 		raise NotImplementedError
#
#
# 	def _update_payload(self, updates):
# 		for key, content in updates.items():
# 			self[key] = content
#
#
# 	def _to(self, device, **kwargs):
# 		updates = {key: content.to(device) for key, content in self._item_iterator()}
# 		self._update_payload(updates)
#
#
# 	def __str__(self):
# 		return f'{self.__class__.__name__}({self._str_info()})'
#
#
#
# class TensorList(TensorContainer, list):
# 	def _str_info(self):
# 		num = len(self)
# 		msg = 'item' if num == 1 else 'items'
# 		return f'{num} {msg}'
#
#
# 	def _item_iterator(self):
# 		return enumerate(self)
#
#
#
# class TensorDict(TensorContainer, dict):
# 	def _str_info(self):
# 		msg = ', '.join(self.keys())
# 		return msg
#
#
# 	def _item_iterator(self):
# 		return self.items()









