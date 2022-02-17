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



class ReconstructionTask(BatchedTask):
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
	def create_results_container(dataset=None, **kwargs):
		return AccumulationContainer(dataset=dataset, **kwargs)
	
	
	@classmethod
	def prepare(cls, dataset=None, encoder=None, decoder=None, criterion=None, **kwargs):
		if encoder is None:
			prt.warning('No encoder provided')
		if decoder is None:
			prt.warning('No decoder provided')
		if criterion is None:
			prt.warning('No criterion provided')
		info = super().prepare(dataset=dataset, **kwargs)
		info.encoder = encoder
		info.decoder = decoder
		info.criterion = criterion
		return info
	
	
	@classmethod
	def run_step(cls, batch, info, slim=None, online=True, gen=None, seed=None):
		info.dataset.change_sample_format(batch, 'original')
		info['original'] = batch
		cls._encode(info)
		cls._decode(info)
		return info

	
	@staticmethod
	def _encode(info):
		info['code'] = info.encoder.encode(info['original'])
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
	def aggregate(cls, info, slim=False, online=False, seed=None, gen=None):
		info = super().aggregate(info)

		criteria = info.aggregate()

		info.update({
			'criteria': criteria,
			'mean': criteria.mean(),
			'max': criteria.max(),
			'min': criteria.min(),
			'std': criteria.std(),
		})
		return info







