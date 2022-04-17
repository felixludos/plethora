import torch
from omnibelt import get_printer, agnosticmethod
from ...framework import models, hparam, inherit_hparams
from ..base import Task, BatchedTask, SimpleEvaluationTask

prt = get_printer(__file__)



class AbstractReconstructionTask(SimpleEvaluationTask):
	@agnosticmethod
	def _compute_step(self, info):
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
		observation = info[self.observation_key]
		info[self.latent_key] = observation if self.encoder is None else self.encoder.encode(observation)
		return info
	
	
	@agnosticmethod
	def _decode_step(self, info):
		info[self.reconstruction_key] = self.decoder.decode(info[self.latent_key])
		return info


	@agnosticmethod
	def _compare_step(self, info):
		info[self.scores_key] = self.criterion.compare(info[self.reconstruction_key], info[self.observation_key])
		return info







