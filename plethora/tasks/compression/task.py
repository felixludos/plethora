import torch
from omnibelt import get_printer, agnosticmethod
from ..base import Task, BatchedTask, SimpleEvaluationTask

prt = get_printer(__file__)


class AbstractCompressionTask(SimpleEvaluationTask):

	@agnosticmethod
	def _compute_step(self, info):
		self._compress(info)
		self._decompress(info)
		return info


	@staticmethod
	def _compress(info):
		raise NotImplementedError


	@agnosticmethod
	def _decompress(self, info):
		raise NotImplementedError



class AbstractLosslessCompressionTask(AbstractCompressionTask):

	strict_verify = False # since the compressor is assumed to be lossless, verifying the invertibility is optional

	@agnosticmethod
	def _compute_step(self, info):
		self._compress(info)
		if self.strict_verify:
			self._decompress(info)
		return info


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








