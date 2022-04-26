import torch
from omnibelt import get_printer, agnosticmethod
from ...framework import hparam, inherit_hparams, models

from .task import AbstractLosslessCompressionTask
from .compressors import LosslessCompressor, get_lossless_compressor
# from .bits_back import BitsBackCompressor



class LosslessCompressionTask(AbstractLosslessCompressionTask):
	reconstruction_key = 'reconstruction'


	compressor_name = hparam('lzma')

	@hparam(cache=True, module=LosslessCompressor)
	def compressor(self):
		return get_lossless_compressor(self.compressor_name)()


	@agnosticmethod
	def _decompress_step(self, info):
		info[self.reconstruction_key] = self.decompress(info[self.bytes_key])
		if not torch.allclose(info[self.reconstruction_key], info[self.observation_key]):
			raise self.DecompressFailed
		return info







