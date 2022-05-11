
import torch
from omnibelt import get_printer, agnosticmethod, unspecified_argument
from ...framework import util, hparam, inherit_hparams, models, util, abstract, math
from ..base import Task, BatchedTask, SimpleEvaluationTask

from .criterion import PathDiscriminator

prt = get_printer(__file__)



class LinearInterpolator(abstract.Interpolator):
	@staticmethod
	def interpolate(start, end, n_steps=12):
		a, b = start.unsqueeze(1), end.unsqueeze(1)
		progress = torch.linspace(0., 1., steps=n_steps, device=a.device).view(1, n_steps, *[1]*len(a.shape[2:]))
		steps = a + (b-a)*progress
		return steps



class AbstractInterpolationTask(SimpleEvaluationTask):
	@agnosticmethod
	def _compute_step(self, info):
		self._pairs_step(info)
		self._interpolate_step(info)
		self._eval_interpolations_step(info)
		return info


	@staticmethod
	def _interpolate_step(info):
		raise NotImplementedError


	@staticmethod
	def _eval_interpolation_step(info):
		raise NotImplementedError



class InterpolationTask(AbstractInterpolationTask, abstract.Interpolator, abstract.PathCriterion):
	observation_key = 'observation'
	endpoint_key = None
	paths_key = 'paths'

	interpolator = hparam(module=abstract.Interpolator)
	path_criterion = hparam(module=abstract.PathCriterion)
	num_steps = hparam(12)
	use_pairwise = hparam(True) # compare all pairs in batch, otherwise just compares half of the batch


	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		if self.endpoint_key is None:
			self.endpoint_key = self.observation_key


	@agnosticmethod
	def interpolate(self, start, end, n_steps=None):
		if n_steps is None:
			n_steps = self.num_steps
		return self.interpolator.interpolate(start, end, n_steps=n_steps)


	@agnosticmethod
	def compare(self, paths, starts, ends):
		return self.path_criterion.compare(paths, starts, ends)


	@agnosticmethod
	def _generate_pairs(self, samples):
		if self.use_pairwise:
			i,j = torch.triu_indices(len(samples), len(samples), offset=1)
			return samples[i], samples[j]
		return samples.chunk(2)


	@agnosticmethod
	def _pairs_step(self, info):
		info[f'{self.endpoint_key}_start'], \
		info[f'{self.endpoint_key}_end'] = self._generate_pairs(info[self.endpoint_key])
		return info


	@agnosticmethod
	def _interpolate_step(self, info):
		info[self.paths_key] = self.interpolate(info[f'{self.endpoint_key}_start'], info[f'{self.endpoint_key}_end'])
		# info['paths'] = info['steps'] if info.decoder is None \
		# 	else util.split_dim(info.decoder.decode(util.combine_dims(info['steps'], 0, 2)), -1, self._num_steps)
		return info


	@agnosticmethod
	def _eval_interpolation_step(self, info):
		info[self.scores_key] = self.compare(info[self.paths_key],
		                                     info[f'{self.endpoint_key}_start'], info[f'{self.endpoint_key}_end'])
		return info
	

	@agnosticmethod
	def aggregate(self, info):
		info = super().aggregate(info)

		scores = info[self.scores_key] # N x num_steps

		info.update({
			'path_mean': scores.mean(0),
			'path_max': scores.max(0)[0],
			'path_min': scores.min(0)[0],
			'path_std': scores.std(0),
		})
		return info



# class LatentInterpolationTask(InterpolationTask, abstract.Encoder, abstract.Decoder):
# 	latent_key = 'latent'
# 	paths_key = 'steps'
# 	paths_key = 'steps'
#
# 	encoder = hparam(module=abstract.Encoder)
# 	decoder = hparam(module=abstract.Decoder)
#
#
# 	def __init__(self, **kwargs):
# 		super().__init__(**kwargs)
# 		if self.endpoint_key is None:
# 			self.endpoint_key = self.latent_key
#
#
# 	@agnosticmethod
# 	def encode(self, observation):
# 		return self.encoder.encode(observation)
#
#
# 	@agnosticmethod
# 	def decode(self, latent):
# 		return self.decoder.decode(latent)
#
#
# 	@agnosticmethod
# 	def _encode_step(self, info):
# 		info[self.latent_key] = self.encode(info[self.observation_key])
# 		return info
#
#
# 	@agnosticmethod
# 	def _decode_step(self, info):
# 		info[self.latent_key] = self.encode(info[self.observation_key])
# 		return info
#
#
# 	@agnosticmethod
# 	def _pairs_step(self, info):
# 		info = self._encode_step(info)
# 		info = super()._pairs_step(info)
# 		return info
#
#
# 	@agnosticmethod
# 	def _interpolate_step(self, info):
# 		info = super()._interpolate_step(info)
# 		info = self._decode_step(info)
# 		return info




