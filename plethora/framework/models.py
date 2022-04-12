

import torch
from omnibelt import agnosticmethod, mix_into
from .features import Seeded
from .base import Function, Container


# class ModelBuilder:
# 	@agnosticmethod
# 	def build(self, dataset):
# 		raise NotImplementedError


class ModelBuilder:
	def __init__(self, **kwargs):
		self.update(**kwargs)


	def update(self, **kwargs):
		self.__dict__.update(kwargs)


	def __call__(self, **kwargs):
		self.update(**kwargs)
		return self.build()


	class MissingKwargsError(Exception):
		def __init__(self, *keys):
			super().__init__(', '.join(keys))
			self.keys = keys


	def build(self):
		raise NotImplementedError



class Resultable:
	score_key = None

	def __init__(self, *args, score_key=None, **kwargs):
		super().__init__(*args, **kwargs)
		if score_key is None:
			self.score_key = score_key


	class ResultsContainer(Seeded, Container):
		def __init__(self, source=None, score_key=None, **kwargs):
			super().__init__(**kwargs)
			self.source = source
			self._score_key = score_key


		class NoScoreKeyError(Exception):
			pass


		def merge_results(self, info):
			self.update(info)


		def _load_missing(self, key, sel=None, **kwargs):
			return self.source.get(key, sel=sel, **kwargs)


		def _find_missing(self, key, **kwargs):
			if key == 'score':
				if self._score_key is None:
					raise self.NoScoreKeyError
				return self[self._score_key]
			if self.source is not None:
				self[key] = self._load_missing(key, **kwargs) # load and cache
				return self[key]
			return super()._find_missing(key)


		def __contains__(self, item):
			return super().__contains__(item) or (item == 'score' and super().__contains__(self._score_key))


	@classmethod
	def _integrate_results(cls, info, **kwargs):
		raise NotImplementedError # TODO
		if not isinstance(info, cls.ResultsContainer):
			new = mix_into(cls.ResultsContainer, info)
		# TODO: run __init__ of new super classes with **kwargs
		return new


	@agnosticmethod
	def create_results_container(self, info=None, score_key=None, **kwargs):
		if score_key is None:
			score_key = self.score_key
		if info is not None:
			return self._integrate_results(info, score_key=score_key, **kwargs)
		return self.ResultsContainer(score_key=score_key, **kwargs)



class Buildable:
	def __init_subclass__(cls, builder=None, **kwargs):
		super().__init_subclass__(**kwargs)
		if builder is None:
			builder = cls.Builder(cls)
		cls.builder = builder


	class Builder(ModelBuilder):
		def __init__(self, cls=None, **kwargs):
			super().__init__(**kwargs)
			if cls is None:
				raise self.MissingSourceClassError
			self.cls = cls


		class MissingSourceClassError(Exception):
			def __init__(self):
				super().__init__('You cannot instantiate a builder without a source class '
				                 '(use cls.builder instead)')


		def build(self):
			kwargs = self.__dict__.copy()
			del kwargs['cls']
			return self.cls(**kwargs)



class Computable(Resultable):
	@agnosticmethod
	def compute(self, source=None, **kwargs):
		info = self.create_results_container(source=source, **kwargs)
		return self._compute(info)


	@staticmethod
	def _compute(info):
		raise NotImplementedError



class Model(Resultable, Buildable):
	@agnosticmethod
	def create_fit_results_container(self, **kwrags):
		return self.create_results_container(**kwargs)


	def fit(self, source, **kwargs):
		info = self.create_fit_results_container(source=source, **kwargs)
		return self._fit(info)


	@staticmethod
	def _fit(info):
		raise NotImplementedError


	def evaluate(self, source, **kwargs):
		info = self.create_results_container(source=source, **kwargs)
		return self._evaluate(info)


	@staticmethod
	def _evaluate(info):
		raise NotImplementedError



class IterativeModel(Model):
	@classmethod
	def create_step_results_container(cls, **kwargs):
		return cls.create_results_container(**kwargs)


	def fit(self, source, info=None, **kwargs):
		# TODO: load default trainer and optimize (wrap this loop with the trainer)
		for batch in source.get_iterator():
			out = self.step(source, info=info, **kwargs)
		return out


	def step(self, source, info=None, **kwargs):
		info = self.create_step_results_container(info=info, **kwargs)
		return self._step(info)


	def _step(self, info):
		raise NotImplementedError



# Types of Models



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


