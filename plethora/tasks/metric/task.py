
import torch
from omnibelt import get_printer, agnosticmethod
from ...datasets import base as dataset_base
from ...framework import util, hparam, inherit_hparams, models
from ..base import Task, BatchedTask, Cumulative

prt = get_printer(__file__)



class AbstractMetricTask(BatchedTask):
	@agnosticmethod
	def _compute_step(self, info):
		self._encode_step(info)
		self._measure_distance(info)
		self._true_distance(info)
		return info


	@staticmethod
	def _encode_step(info):
		raise NotImplementedError


	@staticmethod
	def _measure_distance(info):
		raise NotImplementedError


	@staticmethod
	def _true_distance(info):
		raise NotImplementedError



class MetricTask(AbstractMetricTask):
	score_key = 'agreement'
	observation_key = 'observation'
	latent_key = 'latent'
	mechanism_key = 'mechanism'

	distance_key = 'distance'
	true_distance_key = 'true_distance'


	encoder = hparam(default=None, module=models.Encoder)
	metric = hparam(module=models.Metric)
	criterion = hparam(module=models.Criterion)
	dataset = hparam(module=dataset_base.SyntheticDataset)


	@property
	def encode(self):
		return (lambda x: x) if self.encoder is None else self.encoder.encode


	@property
	def measure(self):
		return self.metric.measure


	@property
	def true_measure(self):
		return self.dataset.distance


	@property
	def compare(self):
		return self.criterion.compare


	@agnosticmethod
	def _encode_step(self, info):
		info[self.latent_key] = self.encode(info[self.observation_key])
		return info


	@staticmethod
	def _split_samples(samples):
		return samples.chunk(2)


	@agnosticmethod
	def _measure_distance(self, info):
		code = info[self.latent_key]
		a, b = self._split_samples(code)
		distance = self.measure(a, b)
		info[self.distance_key] = distance#.squeeze()
		return info


	@agnosticmethod
	def _true_distance(self, info):
		labels = info[self.mechanism_key]
		a, b = self._split_samples(labels)
		info[self.true_distance_key] = self.true_measure(a,b)#.squeeze()
		return info



	class ResultsContainer(Cumulative, BatchedTask.ResultsContainer): # TODO: auto-accumulate scores_key
		def __init__(self, distance_key=None, true_distance_key=None, **kwargs):
			super().__init__(**kwargs)
			if distance_key is not None and true_distance_key is not None:
				self.register_cumulative(distance_key, true_distance_key)


	@agnosticmethod
	def create_results_container(self, info=None, distance_key=None, true_distance_key=None, **kwargs):
		if distance_key is None:
			distance_key = self.distance_key
		if true_distance_key is None:
			true_distance_key = self.true_distance_key
		return super().create_results_container(info=info, distance_key=distance_key,
		                                        true_distance_key=true_distance_key, **kwargs)


	@agnosticmethod
	def aggregate(self, info, slim=False, online=False, seed=None, gen=None):
		info = super().aggregate(info)
		
		distances = info.aggregate(self.distance_key)
		true_distances = info.aggregate(self.true_distance_key)
		agreement = self.compare(distances, true_distances)

		info.update({
			self.distance_key: distances,
			self.true_distance_key: true_distances,
			self.score_key: agreement,
		})
		return info


