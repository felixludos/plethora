import torch
from omnibelt import get_printer, agnosticmethod
from ..base import Task, BatchedTask, SimpleEvaluationTask

from ...framework import hparam, inherit_hparams, models
from .bits_back import BitsBackCompressor

prt = get_printer(__file__)



class AbstractLosslessCompressionTask(SimpleEvaluationTask):

	strict_verify = False

	@agnosticmethod
	def _compute_step(self, info):
		self._compress(info)
		if self.strict_verify:
			self._decompress(info)
		return info


	@staticmethod
	def _compress(info):
		raise NotImplementedError


	class DecompressFailed(Exception):
		pass


	@agnosticmethod
	def _decompress(self, info):
		raise self.DecompressFailed


	@agnosticmethod
	def aggregate(self, info):
		info = super().aggregate(info)
		if not self.strict_verify:
			self._decompress(info)
		return info




class LosslessCompressionTask(AbstractLosslessCompressionTask, SimpleEvaluationTask):
	score_key = 'bpd'

	observation_key = 'observation'
	bytes_key = 'bytes'
	reconstruction_key = 'reconstruction'


	compressor = hparam(module=models.Compressor)


	@agnosticmethod
	def _compress(self, info):
		info[self.bytes_key] = self.compressor.compress(info[self.observation_key])
		info[self.scores_key] = 8 * len(info[self.bytes_key]) / info[self.observation_key].numel()
		return info


	@agnosticmethod
	def _decompress(self, info):
		info[self.reconstruction_key] = self.compressor.decompress(info[self.bytes_key])
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


	@hparam(cache=True)
	def compressor(self):
		return BitsBackCompressor(self.encoder, self.decoder,
		                          beta_confidence=self.beta_confidence, default_scale=self.default_scale)














