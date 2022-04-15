import torch
from omnibelt import get_printer, agnosticmethod
from ..base import Task, BatchedTask, SimpleEvaluationTask

from ...framework import with_hparams, with_args, with_modules, models
from .bits_back import BitsBackCompressor

prt = get_printer(__file__)



@with_args(strict_verify=False)
class AbstractLosslessCompressionTask(SimpleEvaluationTask):
	@agnosticmethod
	def run_step(self, info, **kwargs):
		self._compress(info)
		if info.strict_verify:
			self._decompress(info)
		return info


	@staticmethod
	def _compress(info):
		raise NotImplementedError


	@staticmethod
	def _decompress(info):
		raise NotImplementedError


	@agnosticmethod
	def aggregate(self, info):
		info = super().aggregate(info)
		if not info.strict_verify:
			self._decompress(info)
		return info


	class DecompressFailed(Exception):
		pass



@with_modules(compressor=models.Compressor)
class LosslessCompressionTask(AbstractLosslessCompressionTask):
	score_key = 'bpd'


	@classmethod
	def prepare(cls, encoder=None, decoder=None, compressor=None, decompressor=None,
	            beta_confidence=1000, default_scale=0.1, online=False, verify=None, **kwargs):
		if encoder is None:
			prt.warning('No encoder provided')
		if decoder is None:
			prt.warning('No decoder provided')
		if compressor is None:
			compressor =
		if decompressor is None:
			prt.warning('No decompressor provided')
			decompressor = compressor
		if verify is None:
			verify = not online
		info = super().prepare(online=online, **kwargs)
		info.encoder = encoder
		info.decoder = decoder
		info.compressor = compressor
		info.decompressor = decompressor

		info.verify = verify
		return info


	@classmethod
	def run(cls, info, sample_format=None, **kwargs):
		if sample_format is None:
			sample_format = 'observation'
		return super().run(info, sample_format=sample_format, **kwargs)


	@staticmethod
	def _compress(info):
		if 'original' not in info:
			info['original'] = info.batch
		info['code'] = info.compressor(info['original'])
		info.accumulate(8*len(info['code']) / info['original'].numel())
		return info


	@staticmethod
	def _decompress(info):
		if info.verify:
			info['reconstructions'] = info.decompressor.decompress(info['code'])
			if not torch.allclose(info['reconstructions'], info['original']):
				prt.error('Lossless reconstructions do not match the originals.')
		return info


	@staticmethod
	def aggregate(info, **kwargs):
		info = super().aggregate(info, **kwargs)

		counts = info.aggregate()

		info.update({
			'bpd': counts,
			'mean': counts.mean(),
			'max': counts.max(),
			'min': counts.min(),
			'std': counts.std(),
		})
		return info




@with_modules(compressor=models.Compressor, required=False)
@with_modules(encoder=models.Encoder, decoder=models.Decoder)
@with_hparams(beta_confidence=1000, default_scale=0.1)
class BitsBackCompressionTask(AbstractLosslessCompressionTask):
	@agnosticmethod
	def create_results_container(self, info=None, **kwargs):
		info = super().create_results_container(info=info, **kwargs)
		info.compressor = BitsBackCompressor(info.encoder, info.decoder,
			                                beta_confidence=info.beta_confidence, default_scale=info.default_scale)
		return info





	pass













