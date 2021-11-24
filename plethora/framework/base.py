

from .features import Dimension



class Model(Dimension):
	pass



class Dataset:
	def set_mode(self, mode='train'):
		raise NotImplementedError


	def get_mode(self):
		raise NotImplementedError


	def set_sample_format(self, format):
		raise NotImplementedError


	def get_sample_format(self):
		raise NotImplementedError


	def to_loader(self, infinite=True, sample_format=None, batch_size=60, shuffle=None):
		raise NotImplementedError


	def get_observation_space(self):
		raise NotImplementedError


	def get_observations(self, N=None):
		raise NotImplementedError


	def __len__(self):
		raise NotImplementedError


	def get_samples(self, N=None):
		raise NotImplementedError



class LabeledDataset(Dataset):
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










