
import torch
from omnibelt import get_printer
from ...framework import util
from ..base import Task, BatchedTask

prt = get_printer(__file__)



class AbstractMetricTask(BatchedTask):
	@classmethod
	def run_step(cls, batch, info, **kwargs):
		info.clear()
		info.set_batch(batch)
		cls._encode(info)
		cls._measure_distance(info)
		cls._true_distance(info)
		return info


	@staticmethod
	def _encode(info):
		raise NotImplementedError


	@staticmethod
	def _measure_distance(info):
		raise NotImplementedError


	@staticmethod
	def _true_distance(info):
		raise NotImplementedError



class MetricTask(AbstractMetricTask):
	def __init__(self, encoder=None, metric=None, criterion=None,
	             score_key=None, **kwargs):
		super().__init__(score_key=score_key, **kwargs)

		self.encoder = encoder
		self.metric = metric
		self.criterion = criterion


	@staticmethod
	def create_results_container(dataset=None, **kwargs):
		return AccumulationContainer(dataset=dataset, **kwargs)


	def _compute(self, **kwargs):
		return super()._compute(encoder=self.encoder, metric=self.metric, criterion=self.criterion, **kwargs)


	@classmethod
	def prepare(cls, encoder=None, metric=None, criterion=None, **kwargs):
		if encoder is None:
			prt.warning('No encoder provided')
		if metric is None:
			prt.warning('No metric provided')
		if criterion is None:
			prt.warning('No criterion provided')
		info = super().prepare(**kwargs)
		info.encoder = encoder
		info.metric = metric
		info.criterion = criterion
		return info
	
	
	@classmethod
	def run(cls, info, sample_format=None, **kwargs):
		if sample_format is None:
			sample_format = 'observation', 'mechanism'
		return super().run(info, sample_format=sample_format, **kwargs)

	
	@staticmethod
	def _encode(info):
		if 'original' not in info:
			info['original'] = info.batch[0]
		code = info['original'] if info.encoder is None else info.encoder.encode(info['original'])
		info['code'] = code
		return info


	@staticmethod
	def _measure_distance(info):
		code = info.get('code', info.get('original'))
		a, b = code.split(2)
		distance = info.metric.measure(a, b)
		info.accumulate_distances(distance)
		return info


	@staticmethod
	def _true_distance(info):
		if 'label' not in info:
			info['label'] = info.batch[1]
		a, b = info['label'].split(2)
		distance = info.dataset.distance(a, b)
		info.accumulate_true_distances(distance)
		return info
	

	@classmethod
	def aggregate(cls, info, slim=False, online=False, seed=None, gen=None):
		info = super().aggregate(info)
		
		distances = info.aggregate_distances()
		true_distances = info.aggregate_true_distances()
		agreement = info.criterion(distances, true_distances)

		info.update({
			'distances': distances,
			'true_distances': true_distances,
			
			'score': agreement,
		})
		return info


