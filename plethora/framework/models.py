

import torch
from omnibelt import agnosticmethod, mix_into
from .base import Function


# class ModelBuilder:
# 	@agnosticmethod
# 	def build(self, dataset):
# 		raise NotImplementedError


class ModelBuilder:
	def __init__(self, **kwargs):
		self.__dict__.update(kwargs)


	def __call__(self, **kwargs):
		self.__dict__.update(kwargs)
		return self.build()


	class MissingKwargsError(Exception):
		def __init__(self, *keys):
			super().__init__(', '.join(keys))
			self.keys = keys


	def build(self):
		raise NotImplementedError



class Resultable:
	class ResultsContainer(Seeded, Container):
		def __init__(self, source=None, dataset=None, _skip_super_init=False, **kwargs):
			if not _skip_super_init:
				super().__init__(**kwargs)
				self.dataset = dataset
				if source is None:
					source = dataset
				self.source = source


		def _load_missing(self, key, sel=None, **kwargs):
			return self.source.get(key, sel=sel, **kwargs)


		def _find_missing(self, key, **kwargs):
			if self.source is not None:
				self[key] = self._load_missing(key, **kwargs) # load and cache
				return self[key]
			return super()._find_missing(key)


	def integrate_results(self, info):
		if not isinstance(info, self.ResultsContainer):
			info = mix_into(self.ResultsContainer, info)
		return info


	@classmethod
	def create_results_container(cls, **kwargs):
		return cls.ResultsContainer(**kwargs)



class Buildable:
	@classmethod
	def builder(cls, *args, **kwargs):
		return cls.Builder(*args, cls=cls, **kwargs)

	class Builder(ModelBuilder):
		def __init__(self, cls=None, **kwargs):
			super().__init__(**kwargs)
			if cls is None:
				raise self.MissingSourceClassError
			self.cls = cls

		class MissingSourceClassError(Exception):
			def __init__(self):
				super().__init__('You cannot instantiate a builder without a source class '
				                 '(use builder() instead)')

		def build(self):
			kwargs = self.__dict__.copy()
			del kwargs['cls']
			return self.cls(**kwargs)



class Computable(Resultable):
	def compute(self, **kwargs):
		pass

	def _compute(self, **kwargs):
		raise NotImplementedError



class Fitable(Resultable):
	def fit(self, source):
		raise NotImplementedError



class Model(Resultable, Buildable):



	def evaluate(self, source):
		raise NotImplementedError




class Extractor(Function):
	def extract(self, observation):
		return self(observation)



class Encoder(Extractor):
	def encode(self, observation):
		return self.extract(observation)



class Decoder(Function):
	def decode(self, latent):
		return self(latent)



class Generator(Function):
	def generate(self, N: int, gen=None):
		raise NotImplementedError



class Discriminator(Function):
	def judge(self, observation):
		raise NotImplementedError



class Criterion(Function):
	def compare(self, observation1, observation2):
		raise NotImplementedError



class Metric(Criterion):
	def distance(self, observation1, observation2):
		raise NotImplementedError


	def compare(self, observation1, observation2):
		return self.distance(observation1, observation2)



class Score(Function):
	def score(self, observation):
		raise NotImplementedError



class Interpolator(Function):
	def interpolate(self, start, end, N):
		start, end = start.unsqueeze(1), end.unsqueeze(1)
		progress = torch.linspace(0., 1., steps=N+2, device=start.device).view(1, N+2, *[1] * len(start.shape[2:]))
		return start + (end - start) * progress



class Estimator(Function):
	def predict(self, observation):
		raise NotImplementedError



class Quantizer(Function):
	def quantize(self, observation):
		raise NotImplementedError


	def unquantize(self, observation):
		raise NotImplementedError



class Compressor(Function):
	def compress(self, observation):
		raise NotImplementedError


	def uncompress(self, data):
		raise NotImplementedError



class PathCriterion(Criterion):
	def compare_path(self, path1, path2):
		raise NotImplementedError


