import pickle
import lzma
import numpy as np
import torch
from omnibelt import get_printer, agnosticmethod
from ...framework import hparam, inherit_hparams, models, util, ModuleParametrized



class LosslessCompressor(models.Compressor, ModuleParametrized):
	pass



class SimpleLosslessCompressor(LosslessCompressor):

	@staticmethod
	def _process_tensor(tensor):
		data = tensor.detach().cpu().numpy()
		obj = data.tobytes(), data.dtype, data.shape
		return obj


	@staticmethod
	def _prepare_tensor(obj):
		buffer, dtype, shape = obj
		return torch.from_numpy(np.frombuffer(buffer, dtype=dtype).reshape(*shape))


	@staticmethod
	def _compress_bytes(bytes):
		raise NotImplementedError


	@staticmethod
	def _decompress_bytes(bytes):
		raise NotImplementedError


	@agnosticmethod
	def compress(self, observation):
		return self._compress_bytes(pickle.dumps(self._process_tensor(observation)))


	@agnosticmethod
	def decompress(self, code):
		return self._prepare_tensor(pickle.loads(self._decompress_bytes(code)))



class LZMACompression(SimpleLosslessCompressor):
	@staticmethod
	def _compress_bytes(bytes):
		return lzma.compress(bytes)


	@staticmethod
	def _decompress_bytes(bytes):
		return lzma.decompress(bytes)



lossless_compressor_table = {
	'lzma': LZMACompression,
}
def get_lossless_compressor(ident):
	if not isinstance(ident, str):
		return ident

	if ident in lossless_compressor_table:
		return lossless_compressor_table[ident]
	fixed = ident.replace('-', '').lower()
	if fixed in lossless_compressor_table:
		return lossless_compressor_table[fixed]

	raise NotImplementedError(ident)



class LossyCompressor(models.Compressor, ModuleParametrized):
	pass



class QuantizedCompressor(models.Quantizer, LossyCompressor):

	compressor = hparam(module=models.Compressor)
	quantizer = hparam(module=models.Quantizer)


	@agnosticmethod
	def quantize(self, observation):
		return self.quantizer.quantize(observation)


	@agnosticmethod
	def dequantize(self, observation):
		return self.quantizer.quantize(observation)


	@agnosticmethod
	def compress(self, observation):
		return self.compressor.compress(self.quantize(observation))


	@agnosticmethod
	def decompress(self, data):
		return self.dequantize(self.compressor.decompress(data))



class SigfigQuantizer(models.Quantizer, ModuleParametrized):

	sigfigs = hparam(2)


	@agnosticmethod
	def quantize(self, observation):
		return util.round_sigfigs(observation, self.sigfigs)


	@agnosticmethod
	def dequantize(self, code):
		return util.sigfig_noise(code, torch.rand_like(code) - 0.5, sigfigs=self.sigfigs)



class SigfigCompressor(QuantizedCompressor):

	compressor_name = hparam('lzma')
	sigfigs = hparam(2)


	@hparam(cache=True)
	def quantizer(self):
		return SigfigQuantizer(sigfigs=self.sigfigs)


	@hparam(cache=True)
	def compressor(self):
		return get_lossless_compressor(self.compressor_name)()



lossy_compressor_table = {
	'sigfig': SigfigCompressor,
}
def get_lossy_compressor(ident):
	if not isinstance(ident, str):
		return ident

	if ident in lossy_compressor_table:
		return lossy_compressor_table[ident]
	fixed = ident.replace('-', '').lower()
	if fixed in lossy_compressor_table:
		return lossy_compressor_table[fixed]

	raise NotImplementedError(ident)










