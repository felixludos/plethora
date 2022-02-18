import torch
from omnibelt import get_printer
from ..base import Task, BatchedTask, ResultsContainer

prt = get_printer(__file__)



class AccumulationContainer(ResultsContainer):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.counts = []


	def accumulate(self, counts):
		self.counts.append(counts)


	def aggregate(self):
		return torch.cat(self.counts)



class AbstractLosslessCompressionTask(BatchedTask):
	@classmethod
	def run_step(cls, batch, info, **kwargs):
		info.set_batch(batch)
		cls._compress(info)
		cls._decompress(info)
		return info


	@staticmethod
	def _compress(info):
		raise NotImplementedError


	@staticmethod
	def _decompress(info):
		raise NotImplementedError



class LosslessCompressionTask(AbstractLosslessCompressionTask):
	def __init__(self, compressor=None, encoder=None, decoder=None,
	             sample_format=None, score_key=None, **kwargs):
		if score_key is None:
			score_key = 'bpd'
		if sample_format is None:
			sample_format = 'observation'
		super().__init__(sample_format=sample_format, score_key=score_key, **kwargs)
		self.compressor = compressor
		self.encoder = encoder
		self.decoder = decoder


	@staticmethod
	def score_names():
		return ['mean', 'std', 'min', 'max', *super().score_names()]


	@staticmethod
	def create_results_container(dataset=None, **kwargs):
		return AccumulationContainer(dataset=dataset, **kwargs)


	def _compute(self, **kwargs):
		return super()._compute(encoder=self.encoder, decoder=self.decoder, compressor=self.compressor, **kwargs)


	@classmethod
	def prepare(cls, encoder=None, decoder=None, compressor=None, **kwargs):
		if encoder is None:
			prt.warning('No encoder provided')
		if decoder is None:
			prt.warning('No decoder provided')
		if criterion is None:
			prt.warning('No criterion provided')
		info = super().prepare(**kwargs)
		info.encoder = encoder
		info.decoder = decoder
		info.compressor = compressor
		return info


	@classmethod
	def run(cls, info, sample_format=None, **kwargs):
		if sample_format is None:
			sample_format = 'observation'
		return super().run(info, sample_format=sample_format, **kwargs)


	@staticmethod
	def _compress(info):



		pass


	@staticmethod
	def _decompress(info):
		pass


	@staticmethod
	def aggregate(info, **kwargs):
		info = super().aggregate(info, **kwargs)

		counts = info.aggregate()

		info.update({
			'bpd': counts,
			'mean': criteria.mean(),
			'max': criteria.max(),
			'min': criteria.min(),
			'std': criteria.std(),
		})
		return info











