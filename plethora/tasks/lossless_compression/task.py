import torch
from omnibelt import get_printer
from ..base import Task, BatchedTask, ResultsContainer

from .bits_back import BitsBackCompressor

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
		info.clear()
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
	def __init__(self, compressor=None, decompressor=None, encoder=None, decoder=None,
	             sample_format=None, score_key=None, **kwargs):
		if score_key is None:
			score_key = 'bpd'
		if sample_format is None:
			sample_format = 'observation'
		super().__init__(sample_format=sample_format, score_key=score_key, **kwargs)
		self.compressor = compressor
		self.decompressor = decompressor
		self.encoder = encoder
		self.decoder = decoder


	@staticmethod
	def score_names():
		return ['mean', 'std', 'min', 'max', *super().score_names()]


	@staticmethod
	def create_results_container(dataset=None, **kwargs):
		return AccumulationContainer(dataset=dataset, **kwargs)


	def _compute(self, **kwargs):
		return super()._compute(encoder=self.encoder, decoder=self.decoder,
		                        compressor=self.compressor, decompressor=self.decompressor,
		                        **kwargs)


	@classmethod
	def prepare(cls, encoder=None, decoder=None, compressor=None, decompressor=None,
	            beta_confidence=1000, default_scale=0.1, online=False, verify=None, **kwargs):
		if encoder is None:
			prt.warning('No encoder provided')
		if decoder is None:
			prt.warning('No decoder provided')
		if compressor is None:
			compressor = BitsBackCompressor(encoder, decoder,
			                                beta_confidence=beta_confidence, default_scale=default_scale)
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











