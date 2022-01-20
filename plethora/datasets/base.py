import torch

from omnibelt import unspecified_argument

from ..framework import base

class MissingModeError(Exception):
	def __init__(self):
		super().__init__('')

class DataCollection:
	sample_format = None
	
	def __init__(self, *args, sample_format=unspecified_argument, mode='train', **kwargs):
		super().__init__(*args, **kwargs)
		if sample_format is unspecified_argument:
			sample_format = self.sample_format
		self.set_sample_format(sample_format)
		
		self.mode = mode
		
		self._buffers = {}
		self._modes = {mode: self}
		self._loaded = False


	def register_data(self, name, data=None, space=None, buffer=None):
		if buffer is None:
			buffer = base.Buffer(data=data, space=space)
		self._buffers[name] = buffer
		return self._buffers[name]
	

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
	
	
	def is_loaded(self):
		return self._loaded

	
	def _load(self):
		raise NotImplementedError

	
	def load(self):
		if not self.is_loaded():
			return self._load()


	def __iter__(self):
		self.load()
		return self
	
	
	def __next__(self):
		pass


	
class Dataset(DataCollection):
	def __init__(self, ):
		
		pass
		
		
	def __len__(self):
		raise NotImplementedError
	
	
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
	
	
	def split(self, ratios):
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





