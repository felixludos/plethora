
from omnibelt import unspecified_argument




class Device:
	def __init__(self, *args, device=None, **kwargs):
		super().__init__(*args, **kwargs)
		self.device = device


	def to(self, device, *args, **kwargs):
		self.device = device
		return self._to(device, *args, **kwargs)


	def _to(self, device):
		raise NotImplementedError


class Seeded:
	def __init__(self, gen=None, seed=None, **kwargs):
		raise NotImplementedError
		pass


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


class Loadable(Device):
	def __init__(self, *args, **kwargs): # TODO: add autoload using __certify__
		super().__init__(*args, **kwargs)
		self._loaded = False


	def is_loaded(self):
		return self._loaded


	def load(self, *args, **kwargs):
		if not self.is_loaded():
			self._load(*args, **kwargs)
		return self


	def _load(self, *args, **kwargs):
		raise NotImplementedError



