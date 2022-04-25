import torch
from omnibelt import get_printer, agnosticmethod
from ...framework import hparam, inherit_hparams, models

from .task import AbstractLosslessCompressionTask
from .bits_back import BitsBackCompressor



class LosslessCompressionTask(AbstractLosslessCompressionTask):
	score_key = 'bpd'

	observation_key = 'observation'
	bytes_key = 'bytes'
	reconstruction_key = 'reconstruction'


	compressor = hparam(module=models.Compressor)
	decompressor = hparam(module=models.Decompressor)


	@agnosticmethod
	def compress(self, observation):
		return self.compressor.compress(observation)


	@agnosticmethod
	def decompress(self, data):
		return self.decompressor.decompress(data)


	@agnosticmethod
	def _compress(self, info):
		info[self.bytes_key] = self.compress(info[self.observation_key])
		info[self.scores_key] = 8 * len(info[self.bytes_key]) / info[self.observation_key].numel()
		return info


	@agnosticmethod
	def _decompress(self, info):
		info[self.reconstruction_key] = self.decompress(info[self.bytes_key])
		if not torch.allclose(info[self.reconstruction_key], info[self.observation_key]):
			raise self.DecompressFailed
		return info


	@agnosticmethod
	def aggregate(self, info, **kwargs):
		info = super().aggregate(info, **kwargs)
		info[self.score_key] = 1 - (info['mean'] / 8)
		return info



class BitsBackCompressionTask(LosslessCompressionTask):

	encoder = hparam(module=models.Encoder)
	decoder = hparam(module=models.Decoder)

	beta_confidence = hparam(1000)
	default_scale = hparam(0.1)


	@agnosticmethod
	def encode(self, observation):
		return self.encoder.encode(observation)


	@agnosticmethod
	def decode(self, latent):
		return self.decoder.decode(latent)


	@hparam(cache=True)
	def compressor(self):
		return BitsBackCompressor(self.encode, self.decode,
		                          obs_shape=self.encoder.din.shape, latent_shape=self.decoder.din.shape,
		                          encoder_device=self.encoder.device, decoder_device=self.decoder.device,
		                          beta_confidence=self.beta_confidence, default_scale=self.default_scale)


	@hparam()
	def decompressor(self):
		return self.compressor








