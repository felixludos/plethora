import torch
from ...framework import hparam, inherit_hparams, models
from omnibelt import get_printer, agnosticmethod
from ..base import Task, BatchedTask, SimpleEvaluationTask

prt = get_printer(__file__)


class AbstractCompressionTask(SimpleEvaluationTask):
	score_key = 'mem_score'
	bpd_key = 'bpd'
	scores_key = 'bits_per_dim'

	observation_key = 'observation'
	bytes_key = 'bytes'


	compressor = hparam(module=models.Compressor)


	@agnosticmethod
	def compress(self, observation):
		return self.compressor.compress(observation)


	@agnosticmethod
	def decompress(self, data):
		return self.compressor.decompress(data)


	@agnosticmethod
	def _compute_step(self, info):
		self._compress_step(info)
		self._decompress_step(info)
		return info


	@agnosticmethod
	def compress(self, observation):
		raise NotImplementedError


	@agnosticmethod
	def _compress_step(self, info):
		info[self.bytes_key] = self.compress(info[self.observation_key])
		info[self.scores_key] = 8 * len(info[self.bytes_key]) / info[self.observation_key].numel()
		return info


	@agnosticmethod
	def _decompress_step(self, info):
		raise NotImplementedError


	def aggregate(self, info):
		info = super().aggregate(info)
		info[self.bpd_key] = info['mean']
		info[self.score_key] = 1. - min(max(0., info[self.bpd_key] / 8.), 1.)
		return info



class AbstractLosslessCompressionTask(AbstractCompressionTask):

	strict_verify = False # since the compressor is assumed to be lossless, verifying the invertibility is optional

	@agnosticmethod
	def _compute_step(self, info):
		self._compress_step(info)
		if self.strict_verify:
			self._decompress_step(info)
		return info


	class DecompressFailed(Exception):
		pass


	@agnosticmethod
	def _decompress_step(self, info):
		raise self.DecompressFailed


	@agnosticmethod
	def aggregate(self, info):
		info = super().aggregate(info)
		if not self.strict_verify:
			self._decompress_step(info)
		return info








