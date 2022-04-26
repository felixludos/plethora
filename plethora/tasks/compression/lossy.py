import lzma
import torch
from omnibelt import get_printer, agnosticmethod
from ...framework import hparam, inherit_hparams, models

from ..reconstruction import get_criterion
from .compressors import SigfigCompressor, LossyCompressor
from .task import AbstractCompressionTask



class LossyCompressionTask(AbstractCompressionTask):
	score_key = 'fidelity' # harmonic mean between the reconstruction [0-1] (eg. MS-SSIM)
							# and the bits/dim [0-1] (relative to uncompressed)
	mem_score_key = 'mem_score'
	img_score_key = 'img_score'

	img_scores_key = 'img_scores'

	reconstruction_key = 'reconstruction'


	memory_weight = hparam(1.) # weight for the compression component of the score (analogous to beta in F_beta-score)

	compressor = hparam(module=LossyCompressor)
	criterion_name = hparam('ms-ssim')


	@hparam(cache=True, module=models.Criterion)
	def criterion(self):
		return get_criterion(self.criterion_name)()


	@agnosticmethod
	def compress(self, observation):
		return self.compressor.compress(observation)


	@agnosticmethod
	def decompress(self, data):
		return self.compressor.decompress(data)


	@agnosticmethod
	def compare(self, observation, reconstruction):
		return self.criterion.compare(observation, reconstruction)


	@agnosticmethod
	def _decompress_step(self, info):
		info[self.reconstruction_key] = self.decompress(info[self.bytes_key])
		info[self.img_scores_key] = self.compare(info[self.reconstruction_key], info[self.observation_key])
		return info


	class ResultsContainer(AbstractCompressionTask.ResultsContainer): # TODO: auto-accumulate scores_key
		def __init__(self, img_scores_key=None, **kwargs):
			super().__init__(**kwargs)
			if img_scores_key is not None:
				self.register_cumulative(img_scores_key)


	@agnosticmethod
	def create_results_container(self, info=None, img_scores_key=None, **kwargs):
		if img_scores_key is None:
			img_scores_key = self.img_scores_key
		return super().create_results_container(info=info, img_scores_key=img_scores_key, **kwargs)


	@agnosticmethod
	def aggregate(self, info):
		info = super().aggregate(info)

		img_scores = info.aggregate(self.img_scores_key)
		info.update({
			f'full_{self.img_scores_key}': img_scores,
			'img_score_mean': img_scores.mean().item(),
			'img_score_max': img_scores.max().item(),
			'img_score_min': img_scores.min().item(),
			'img_score_mstd': img_scores.std().item(),

			'bpd_mean': info['mean'],
			'bpd_max': info['max'],
			'bpd_min': info['min'],
			'bpd_std': info['std'],
		})
		del info['mean'], info['max'], info['min'], info['std']

		info[self.mem_score_key] = info[self.score_key] # from super()
		info[self.img_score_key] = info['img_score_mean']

		img, mem = info[self.img_score_key], info[self.mem_score_key]
		info[self.score_key] = (1 + self.memory_weight**2) * mem * img / (mem + self.memory_weight**2 * img)
		return info


	@agnosticmethod
	def score_names(self):
		prev = super().score_names()
		prev.difference_update({'mean', 'std', 'min', 'max'})
		return {'bpd_mean', 'bpd_max', 'bpd_min', 'bpd_std', self.mem_score_key,
		        'img_score_mean', 'img_score_max', 'img_score_min', 'img_score_std',
		        *prev}


	@agnosticmethod
	def heavy_results(self):
		return {f'full_{self.img_scores_key}', *super().heavy_results()}



@inherit_hparams('criterion', 'criterion_name', 'memory_weight')
class RoundingCompressionTask(LossyCompressionTask):

	compressor_name = hparam('lzma')
	sigfigs = hparam(2)

	encoder = hparam(module=models.Encoder)
	decoder = hparam(module=models.Decoder)


	@hparam(module=LossyCompressor)
	def compressor(self):
		return SigfigCompressor(compressor_name=self.compressor_name, sigfigs=self.sigfigs)


	@agnosticmethod
	def encode(self, observation):
		return self.encoder.encode(observation)


	@agnosticmethod
	def compress(self, observation):
		return super().compress(self.encode(observation))


	@agnosticmethod
	def decode(self, latent):
		return self.decoder.decode(latent)


	@agnosticmethod
	def decompress(self, data):
		return self.decode(super().decompress(data))





