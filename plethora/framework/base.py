
from omnibelt import unspecified_argument, duplicate_instance

from .features import Device, Loadable



class BufferTransform:
	def transform(self, sample=None):
		return sample


class BufferUpdateFailedError(NotImplementedError):
	pass


class Buffer(BufferTransform, Loadable, Device):
	space = None
	
	def __init__(self, space=unspecified_argument, **kwargs):
		super().__init__(**kwargs)
		if space is unspecified_argument:
			space = self.space
		self.space = space
		self._collection = None
		self._updated_idx = None
		self._loaded = False


	def copy(self):
		return duplicate_instance(self) # shallow copy


	def set_collection(self, collection):
		self._collection = collection


	def set_space(self, space):
		self.space = space


	def get_space(self):
		return self.space

	def load(self, *args, **kwargs):
		if not self.is_loaded():
			out = self._load(idx=idx, **kwargs)
			if self._updated_idx is not None:
				update = self._updated_idx
				self._updated_idx = None
				self.update(self._updated_idx)


	# def _load(self, idx=None, **kwargs):
	# 	raise NotImplementedError


	def _update(self, idx, *args, **kwargs):
		raise BufferUpdateFailedError


	def update(self, idx, *args, **kwargs):
		try:
			self._update(idx, *args, **kwargs)
		except BufferUpdateFailedError:
			if self._updated_idx is None:
				self._updated_idx = idx
			else:
				self._updated_idx = self._updated_idx[idx]


	def _get(self, idx, *args, **kwargs):
		raise NotImplementedError


	def get(self, idx, *args, **kwargs):
		sample = self._get(idx, *args, **kwargs)
		return self.transform(sample)


	def __getitem__(self, idx):
		return self.get(idx)



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
	










