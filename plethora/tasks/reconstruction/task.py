
from ..base import Task, BatchedTask



class BatchAccumulationContainer(TensorDict):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.criteria = []

	def gen_batch(self):
		raise NotImplementedError

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


	def _create_batch_results_container(self, batch):
		return BatchAccumulationContainer(batch)


	def _compute_batch(self, info, slim=None, gen=None, seed=None):
		self._encode(info)
		self._decode(info)
		return info


	def _encode(self, info):
		original = info.batch

		code = self.encoder.encode(original)
		info.update({
			'original': original,
			'code': code,
		})
		return info


	def _decode(self, info):
		code = info['code']

		reconstruction = self.decoder.decode(code)
		comparison = self.criterion.compare(reconstruction, original)

		info.accumulate(comparison)
		info.update({
			'reconstruction': reconstruction,
			'comparison': comparison,
		})
		return info


	@staticmethod
	def _aggregate(self, overall_info, batch_info):
		overall_info = super()._aggregate(overall_info, batch_info)

		criteria = batch_info.aggregate()

		overall_info.update({
			'criteria': criteria,
			'mean': criteria.mean(),
			'max': criteria.max(),
			'min': criteria.min(),
			'std': criteria.std(),
		})
		return overall_info







