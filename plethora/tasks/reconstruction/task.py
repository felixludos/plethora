import torch
from omnibelt import get_printer
from ..base import Task, BatchedTask, ResultsContainer

prt = get_printer(__file__)



class AccumulationContainer(ResultsContainer):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.criteria = []


	def accumulate(self, criteria):
		self.criteria.append(criteria)


	def aggregate(self):
		return torch.cat(self.criteria)



class AbstractReconstructionTask(BatchedTask):
	@classmethod
	def run_step(cls, batch, info, **kwargs):
		info.clear()
		info.set_batch(batch)
		cls._encode(info)
		cls._decode(info)
		return info


	@staticmethod
	def _encode(info):
		raise NotImplementedError


	@staticmethod
	def _decode(info):
		raise NotImplementedError



class ReconstructionTask(AbstractReconstructionTask):
	def __init__(self, encoder=None, decoder=None, criterion=None,
	             sample_format=None, score_key=None, **kwargs):
		if score_key is None:
			score_key = 'mean'
		if sample_format is None:
			sample_format = 'observation'
		super().__init__(sample_format=sample_format, score_key=score_key, **kwargs)
		self.encoder = encoder
		self.decoder = decoder
		self.criterion = criterion


	@staticmethod
	def score_names():
		return ['mean', 'std', 'min', 'max', *super().score_names()]


	@staticmethod
	def create_results_container(dataset=None, **kwargs):
		return AccumulationContainer(dataset=dataset, **kwargs)


	def _compute(self, **kwargs):
		return super()._compute(encoder=self.encoder, decoder=self.decoder, criterion=self.criterion, **kwargs)


	@classmethod
	def prepare(cls, encoder=None, decoder=None, criterion=None, **kwargs):
		if encoder is None:
			prt.warning('No encoder provided')
		if decoder is None:
			prt.warning('No decoder provided')
		if criterion is None:
			prt.warning('No criterion provided')
		info = super().prepare(**kwargs)
		info.encoder = encoder
		info.decoder = decoder
		info.criterion = criterion
		return info


	@classmethod
	def run(cls, info, sample_format=None, **kwargs):
		if sample_format is None:
			sample_format = 'observation'
		return super().run(info, sample_format=sample_format, **kwargs)

	
	@staticmethod
	def _encode(info):
		code = info['original'] if info.encoder is None else info.encoder.encode(info['original'])
		info['code'] = code
		return info


	@staticmethod
	def _decode(info):
		original = info['original']
		code = info['code']

		reconstruction = info.decoder.decode(code)
		comparison = info.criterion.compare(reconstruction, original)

		info.accumulate(comparison)
		info.update({
			'reconstruction': reconstruction,
			'comparison': comparison,
		})
		return info


	@classmethod
	def aggregate(cls, info, **kwargs):
		info = super().aggregate(info, **kwargs)

		criteria = info.aggregate()

		info.update({
			'criteria': criteria,
			'mean': criteria.mean(),
			'max': criteria.max(),
			'min': criteria.min(),
			'std': criteria.std(),
		})
		return info







