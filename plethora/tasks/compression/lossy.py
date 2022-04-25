import torch
from omnibelt import get_printer, agnosticmethod
from ...framework import hparam, inherit_hparams, models

from .task import AbstractCompressionTask




class LossyCompressionTask(AbstractCompressionTask):
	score_key = 'fidelity' # harmonic mean between the reconstruction [0-1] (eg. MS-SSIM)
							# and the bits/dim [0-1] (relative to uncompressed)

	observation_key = 'observation'
	bytes_key = 'bytes'
	reconstruction_key = 'reconstruction'

	memory_weight = hparam(1.) # weight for the compression component of the score (analogous to beta in F_beta-score)

	compressor = hparam(module=models.Compressor)
	decompressor = hparam(module=models.Decompressor)
	criterion = hparam(module=models.Criterion)


	@agnosticmethod
	def compress(self, observation):
		return self.compressor.compress(observation)


	@agnosticmethod
	def decompress(self, data):
		return self.decompressor.decompress(data)


	pass



class RoundingCompressionTask(LossyCompressionTask):

	pass

