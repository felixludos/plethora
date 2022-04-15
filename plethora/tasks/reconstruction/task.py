import torch
from omnibelt import get_printer, agnosticmethod
from ...framework import with_hparams, with_modules, models
from ..base import Task, BatchedTask, SimpleEvaluationTask

prt = get_printer(__file__)



class AbstractReconstructionTask(SimpleEvaluationTask):
	@agnosticmethod
	def run_step(self, info):
		self._encode_step(info)
		self._decode_step(info)
		self._compare_step(info)
		return info

	observation_key = 'observation'

	@staticmethod
	def _encode_step(info):
		raise NotImplementedError


	@staticmethod
	def _decode_step(info):
		raise NotImplementedError

	
	@staticmethod
	def _compare_step(info):
		raise NotImplementedError



@with_modules(encoder=models.Encoder, required=False)
@with_modules(decoder=models.Decoder, criterion=models.Criterion, required=True)
class ReconstructionTask(AbstractReconstructionTask):
	latent_key = 'latent'
	reconstruction_key = 'reconstruction'

	@agnosticmethod
	def _encode_step(self, info, *, observation=None):
		if observation is None:
			observation = info[self.observation_key]
		info[self.latent_key] = observation if info.encoder is None else info.encoder.encode(observation)
		return info


	@agnosticmethod
	def _decode_step(self, info, *, latent=None):
		if latent is None:
			latent = info[self.latent_key]
		info[self.reconstruction_key] = info.decoder.decode(latent)
		return info


	@agnosticmethod
	def _compare_step(self, info, *, original=None, reconstruction=None):
		if original is None:
			original = info[self.observation_key]
		if reconstruction is None:
			reconstruction = info[self.reconstruction_key]
		info[info.evaluation_key] = info.criterion.compare(reconstruction, original)
		return info






