import torch
from omnibelt import get_printer
from ..base import Task, BatchedTask, SimpleEvaluationTask

prt = get_printer(__file__)



class AbstractReconstructionTask(SimpleEvaluationTask):
	@classmethod
	def _eval_step(cls, info, *, comparison_key='comparison'):
		cls._encode_step(info)
		cls._decode_step(info)
		cls._compare_step(info, comparison_key=comparison_key)
		return info


	@staticmethod
	def _encode_step(info):
		raise NotImplementedError


	@staticmethod
	def _decode_step(info):
		raise NotImplementedError

	
	@staticmethod
	def _compare_step(info, *, comparison_key='comparison'):
		raise NotImplementedError



class ReconstructionTask(AbstractReconstructionTask):
	def __init__(self, encoder=None, decoder=None, criterion=None, **kwargs):
		super().__init__(**kwargs)
		self.encoder = encoder
		self.decoder = decoder
		self.criterion = criterion


	def prepare(self, encoder=None, decoder=None, criterion=None, **kwargs):
		if encoder is None:
			encoder = self.encoder
		if decoder is None:
			decoder = self.decoder
		if criterion is None:
			criterion = self.criterion
		
		if encoder is None:
			prt.warning('No encoder provided') # TODO: check required and fill in defaults
		if decoder is None:
			prt.warning('No decoder provided')
		if criterion is None:
			prt.warning('No criterion provided')
		
		info = super().prepare(**kwargs)
		info.encoder = encoder
		info.decoder = decoder
		info.criterion = criterion
		return info

	
	@staticmethod
	def _encode_step(info, *, original=None, code_key='code'):
		if original is None:
			original = info['observation']
		info[code_key] = original if info.encoder is None else info.encoder.encode(original)
		return info


	@staticmethod
	def _decode_step(info, *, code=None, reconstruction_key='reconstruction'):
		if code is None:
			code = info['code']
		info[reconstruction_key] = info.decoder.decode(code)
		return info


	@staticmethod
	def _compare_step(info, *, original=None, reconstruction=None, comparison_key='comparison'):
		if original is None:
			original = info['observation']
		if reconstruction is None:
			reconstruction = info['reconstruction']
		info[comparison_key] = info.criterion.compare(reconstruction, original)
		return info






