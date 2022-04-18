

import torch
from omnibelt import agnosticmethod, unspecified_argument#, mix_into
from .features import Seeded
from .hyperparameters import Parametrized, ModuleParametrized, hparam, inherit_hparams
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


	class MissingKwargsError(KeyError):
		def __init__(self, *keys):
			super().__init__(', '.join(keys))
			self.keys = keys


	def build(self):
		raise NotImplementedError



class Resultable:
	seed = None
	score_key = None

	class ResultsContainer(Seeded, Container):
		def __init__(self, source=None, score_key=None, **kwargs):
			super().__init__(**kwargs)
			self.source = source
			self._score_key = score_key


		def new_source(self, source):
			self.clear()
			self.source = source


		class NoScoreKeyError(Exception):
			pass


		def merge_results(self, info):
			self.update(info)


		def _load_missing(self, key, **kwargs):
			return self.source.get(key, **kwargs)


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


	@agnosticmethod
	def _integrate_results(self, info, **kwargs):
		raise NotImplementedError # TODO
		if not isinstance(info, self.ResultsContainer):
			new = mix_into(self.ResultsContainer, info)
		# TODO: run __init__ of new super classes with **kwargs
		return new


	@agnosticmethod
	def create_results_container(self, info=None, score_key=None, seed=unspecified_argument, **kwargs):
		if score_key is None:
			score_key = self.score_key
		if seed is unspecified_argument:
			seed = self.seed
		if info is not None:
			return self._integrate_results(info, score_key=score_key, seed=seed, **kwargs)
		return self.ResultsContainer(score_key=score_key, **kwargs)



class Buildable(ModuleParametrized): # TODO: unify building and hparams - they should be shared
	# def __init_subclass__(cls, builder=None, **kwargs):
	# 	super().__init_subclass__(**kwargs)
	# 	if builder is None:
	# 		builder = cls.Builder(cls)
	# 	cls.builder = builder

	# _my_build_settings = {} # in a sub-class of Buildable


	@agnosticmethod
	def get_builder(self, cls=None, **kwargs):
		if cls is None:
			cls = self if isinstance(self, type) else self.__class__
		return self.Builder(cls=cls, **kwargs)


	class Builder(ModelBuilder):
		def __init__(self, cls=None, **kwargs):
			super().__init__(**kwargs)
			if cls is None:
				raise self.MissingSourceClassError
			self.update(**{key:getattr(cls, key) for key in cls.iterate_hparams()})
			self.cls = cls


		class MissingSourceClassError(Exception):
			def __init__(self):
				super().__init__('You cannot instantiate a builder without a source class '
				                 '(use cls.builder instead)')


		def build(self, kwargs=None):
			if kwargs is None:
				kwargs = self.__dict__.copy()
				del kwargs['cls']
			return self.cls(**kwargs)



class Computable(ModuleParametrized, Resultable):
	@agnosticmethod
	def compute(self, source=None, **kwargs):
		info = self.create_results_container(source=source, **kwargs)
		return self._compute(info)


	@staticmethod
	def _compute(info):
		raise NotImplementedError



class Fitable(Resultable):
	def fit(self, source, **kwargs):
		raise NotImplementedError


	def evaluate(self, source, **kwargs):
		raise NotImplementedError



class Model(Buildable, Fitable):

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



class Trainer(ModuleParametrized, Fitable):
	model = hparam(module=Model)

	def __init__(self, model, source=None, **kwargs):
		super().__init__(**kwargs)
		self.source = source
		self.model = model

		self.N_iter = 0
		self.N_samples = 0


	def loop(self, source, **kwargs):
		itr = source.get_iterator(**kwargs)
		for batch in itr:
			yield batch
			self.N_iter += 1
			self.N_samples += batch.size


	@agnosticmethod
	def create_step_results_container(self, **kwargs):
		return self.model.create_step_results_container(**kwargs)


	def fit(self, source=None, **kwargs):
		if source is None:
			source = self.source
		for batch in self.loop(source, **kwargs):
			info = self.step(batch)
		return self.finish_fit(info)


	def evaluate(self, source=None, **kwargs):
		if source is None:
			source = self.source
		info = self.model.evaluate(source, **kwargs)
		return self.finish_evaluate(info)


	def step(self, source, **kwargs):
		info = self.create_step_results_container(source=source, **kwargs)
		return self._step(info)


	def _step(self, info, **kwargs):
		return self.model.step(info)


	def finish_fit(self, info):
		return info


	def finish_evaluate(self, info):
		return info



class TrainableModel(Model):
	@agnosticmethod
	def create_step_results_container(self, **kwargs):
		return self.create_results_container(**kwargs)


	Trainer = Trainer
	@agnosticmethod
	def fit(self, source, info=None, **kwargs):
		assert info is None, 'cant merge info'
		trainer = self.Trainer(self)
		return trainer.fit(source=source, **kwargs)


	@agnosticmethod
	def step(self, info, **kwargs):
		self._step(info, **kwargs)
		return info


	@agnosticmethod
	def _step(self, info):
		raise NotImplementedError



# Types of Models



class Extractor(Function):
	@agnosticmethod
	def extract(self, observation):
		raise NotImplementedError



class Encoder(Extractor):
	@agnosticmethod
	def encode(self, observation):
		raise NotImplementedError



class Decoder(Function):
	@agnosticmethod
	def decode(self, latent):
		raise NotImplementedError



class Generator(Function):
	@agnosticmethod
	def generate(self, N: int, gen=None):
		raise NotImplementedError



class Discriminator(Function):
	@agnosticmethod
	def judge(self, observation):
		raise NotImplementedError



class Criterion(Function):
	@agnosticmethod
	def compare(self, observation1, observation2):
		raise NotImplementedError



class Metric(Criterion): # obeys triangle inequality
	@agnosticmethod
	def distance(self, observation1, observation2):
		raise NotImplementedError


	@agnosticmethod
	def compare(self, observation1, observation2):
		return self.distance(observation1, observation2)



# class Score(Function):
# 	@agnosticmethod
# 	def score(self, observation):
# 		raise NotImplementedError



class Interpolator(Function): # returns N steps to get from start to finish ("evenly spaces", by default)
	@staticmethod
	def interpolate(start, end, N):
		start, end = start.unsqueeze(1), end.unsqueeze(1)
		progress = torch.linspace(0., 1., steps=N+2, device=start.device).view(1, N+2, *[1] * len(start.shape[2:]))
		return start + (end - start) * progress



class Estimator(Function):
	@agnosticmethod
	def predict(self, observation):
		raise NotImplementedError



class Quantizer(Function):
	@agnosticmethod
	def quantize(self, observation):
		raise NotImplementedError


	@agnosticmethod
	def unquantize(self, observation):
		raise NotImplementedError



class Compressor(Function):
	@staticmethod
	def compress(observation):
		raise NotImplementedError


	@staticmethod
	def decompress(data):
		raise NotImplementedError



class PathCriterion(Criterion):
	@agnosticmethod
	def compare_path(self, path1, path2):
		raise NotImplementedError


