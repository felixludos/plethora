import torch
from omnibelt import get_printer, agnosticmethod
from ...framework import models, hparam, inherit_hparams
from ..base import Task, BatchedTask, SimpleEvaluationTask

prt = get_printer(__file__)



class AbstractReconstructionTask(SimpleEvaluationTask):
	@agnosticmethod
	def run_step(self, info):
		self._encode_step(info)
		self._decode_step(info)
		self._compare_step(info)
		return info


	@staticmethod
	def _encode_step(info):
		raise NotImplementedError


	@staticmethod
	def _decode_step(info):
		raise NotImplementedError

	
	@staticmethod
	def _compare_step(info):
		raise NotImplementedError



class ReconstructionTask(AbstractReconstructionTask):
	observation_key = 'observation'
	latent_key = 'latent'
	reconstruction_key = 'reconstruction'

	
	encoder = hparam(default=None, module=models.Encoder)
	decoder = hparam(module=models.Decoder)
	criterion = hparam(module=models.Criterion)


	@agnosticmethod
	def _encode_step(self, info):
		info[self.latent_key] = self.encode(info[self.observation_key])
		return info

	
	@agnosticmethod
	def encode(self, observation):
		return observation if self.encoder is None else self.encoder.encode(observation)
	
	
	@agnosticmethod
	def _decode_step(self, info):
		info[self.reconstruction_key] = self.decode(info[self.latent_key])
		return info
	
	
	@agnosticmethod
	def decode(self, latent):
		return self.decoder.decode(latent)


	@agnosticmethod
	def _compare_step(self, info):
		info[self.evaluation_key] = self.compare(info[self.reconstruction_key], info[self.observation_key])
		return info

	
	@agnosticmethod
	def compare(self, reconstruction, original):
		return self.criterion.compare(reconstruction, original)






