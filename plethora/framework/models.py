

import torch

from .base import Function


class Encoder(Function):
	def encode(self, observation):
		return self(observation)



class Decoder(Function):
	def decode(self, latent):
		return self(latent)



class Generator(Function):
	def generate(self, N: int):
		raise NotImplementedError



class Discriminator(Function):
	def judge(self, observation):
		raise NotImplementedError



class Extractor(Encoder):
	def extract(self, observation):
		return self.encode(observation)



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


