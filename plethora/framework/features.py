import os
from collections import OrderedDict
from pathlib import Path
import torch
from omnibelt import unspecified_argument, agnosticmethod

from . import util



class Named:
	def __init__(self, name=unspecified_argument, **kwargs):
		if name is not unspecified_argument:
			self.name = name


	def __str__(self):
		return self.name


	@property
	def name(self):
		try:
			return self._name
		except AttributeError:
			return getattr(self.__class__, 'name', self.__class__.__name__)
	@name.setter
	def name(self, name):
		self._name = name



class Device:
	def __init__(self, *args, device=None, **kwargs):
		super().__init__(*args, **kwargs)
		self.device = device


	def to(self, device, **kwargs):
		self.device = device
		out = self._to(device, **kwargs)
		if out is not None:
			return out
		return self


	def _to(self, device, **kwargs):
		raise NotImplementedError



class DeviceContainer(Device):
	def __init__(self, children={}, **kwargs):
		super().__init__(**kwargs)
		self._device_children = set()
		self._register_deviced_children(**children)


	def _register_deviced_children(self, **children):
		for name, child in children.items():
			self._device_children.add(name)
			setattr(self, name, child)


	def _to(self, device, **kwargs):
		children = {}
		for name in self._device_children:
			obj = getattr(self, name, None)
			if obj is not None:
				children[name] = obj.to(device)
		self._register_deviced_children(**children)



class Seeded:
	def __init__(self, gen=None, seed=None, **kwargs):
		super().__init__(**kwargs)
		self.seed = seed
		self.gen = gen
		if gen is None:
			self.set_seed(self.seed)


	@staticmethod
	def create_rng(gen=None, seed=None):
		if seed is None:
			seed = util.gen_random_seed(gen)
		gen = torch.Generator()
		gen.manual_seed(seed)
		return gen


	def set_seed(self, seed=None):
		self.gen = self.create_rng(gen=self.gen, seed=seed)



class SharableAttrs:
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self._shared_attrs = set()


	def _make_shared(self, key, val=unspecified_argument):
		if val is unspecified_argument:
			val = getattr(self, key)
		self._shared_attrs.add(key)
		setattr(self, key, [val])


	def access_shared(self, key):
		val = getattr(self, key, None)
		if val is not None and key in self._shared_attrs:
			return val[0]
		return val



class Prepared: # TODO: add autoprepare using __certify__
	_is_ready = False

	@property
	def is_ready(self):
		return self._is_ready


	class NotReady(Exception):
		pass


	def prepare(self, *args, **kwargs):
		if not self.is_ready:
			self._prepare(*args, **kwargs)
			self._is_ready = True
		return self


	def _prepare(self, *args, **kwargs):
		raise NotImplementedError



class Rooted:
	_root = None
	def __init__(self, root=unspecified_argument, **kwargs):
		super().__init__(**kwargs)
		if root is not unspecified_argument:
			self._root = root


	@staticmethod
	def _infer_root(root=None):
		if root is None:
			root = os.getenv('PLETHORA_PATH', 'local_data/')
		root = Path(root)
		os.makedirs(str(root), exist_ok=True)
		return root


	@agnosticmethod
	def get_root(self): # for easier inheritance
		return self._infer_root(self._root)


	@property
	def root(self):
		return self.get_root()




# class Downloadable: # TODO
# 	pass


