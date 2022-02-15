
import torch

from ..framework.features import Seeded
from ..framework.base import TensorDict



class ResultsContainer(TensorDict):
	def __init__(self, dataset, **kwargs):
		super().__init__(**kwargs)
		self.dataset = dataset



class Task(Seeded): # TODO: specify random seed for reproducibility
	def __init__(self, dataset=None, slim=False, online=False, score_key=None, **kwargs):
		super().__init__(**kwargs)
		self.dataset = dataset
		self._slim = slim
		self._online = online
		self._score_key = score_key


	def score_names(self):
		return ['score']


	def result_names(self):
		return []


	def _create_results_container(self):
		return ResultsContainer(self.dataset)


	def compute(self, slim=None, online=None, seed=None, gen=None, info=None):
		'''
		:param slim: use little space (in results)
		:param online: use little time (in computing results)
		:param seed: random seed
		:param gen: torch.Generator instance (defaults to self.gen)
		:param info: results container (created using self._create_results_container)
		:return: results container
		'''
		if slim is None:
			slim = self._slim
		if online is None:
			online = self._online
		if seed is not None:
			gen = torch.Generator().manual_seed(seed)
		if gen is None:
			gen = self.gen

		if info is None:
			info = self._create_results_container()
		info = self._compute(info=info, slim=slim, online=online, gen=gen, seed=seed)

		# if slim: # TODO: auto_slim
		# 	info = self._slim_down(info)
		# return info


	# def _slim_down(self, info):
	# 	for key in list(keys)


	def _compute(self, info, slim=False, online=False, gen=None, seed=None):
		raise NotImplementedError



class BatchResultsContainer(TensorDict):
	def __init__(self, batch, **kwargs):
		super().__init__(**kwargs)
		self.set_batch(batch)

	def set_batch(self, batch):
		self.batch = batch



class BatchedTask(Task):
	def __init__(self, sample_format=None, device=None, drop_last=None, batch_size=None, shuffle=None, **kwargs):
		super().__init__(**kwargs)
		self._sample_format = sample_format
		self._batch_size = batch_size
		self._drop_last = drop_last


	def _create_batch_results_container(self, batch):
		return BatchResultsContainer(batch)


	def _compute(self, overall, slim=False, online=False, gen=None, seed=None):
		dataset = overall.dataset

		if seed is not None:
			gen = torch.Generator().manual_seed(seed)
		if gen is None:
			gen = self.gen

		info = None
		for batch in dataset.get_iterator(sample_format=self._sample_format, batch_size=self._batch_size,
		                                  shuffle=True, drop_last=self._drop_last, gen=gen):
			if info is None:
				info = self._create_batch_results_container(batch)
			else:
				info.set_batch(batch)
			info = self._compute_batch(info, slim=slim, online=online, gen=gen)

			if online:
				break

		return self._aggregate(overall, info, slim=slim, online=online, gen=gen)


	def _aggregate(self, overall_info, batch_info, slim=None, online=None, gen=None):
		overall_info.update(batch_info)
		return overall_info


	def _compute_batch(self, info, slim=False, online=False, gen=None):
		raise NotImplementedError






