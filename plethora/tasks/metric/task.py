
import torch
from torch.nn import functional as F
from omnibelt import get_printer, agnosticmethod
from scipy import stats

from ...datasets import base as dataset_base
from ...framework import util, hparam, inherit_hparams, models, util, abstract, math
from ..base import Task, BatchedTask, Cumulative, SimpleEvaluationTask

from .metrics import get_metric

prt = get_printer(__file__)



class AbstractMetricTask(SimpleEvaluationTask):
	@agnosticmethod
	def _compute_step(self, info):
		self._encode_step(info)
		self._measure_distance(info)
		self._true_distance(info)
		self._compare_distances(info)
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


	@staticmethod
	def _compare_distances(info):
		raise NotImplementedError



class MetricTask(AbstractMetricTask):
	scores_key = 'errors'

	observation_key = 'observation'
	latent_key = 'latent'
	target_key = 'target'

	distance_key = 'distance'
	true_distance_key = 'true_distance'

	encoder = hparam(default=None, module=models.Encoder)
	metric = hparam(module=abstract.Metric)
	criterion = hparam(module=models.Criterion)
	dataset = hparam(module=dataset_base.SupervisedDataset) # need some supervision for true distances
	use_pairwise = hparam(True) # compare all pairs in batch, otherwise just compares half of the batch
	standardize_true = hparam(True)

	@agnosticmethod
	def encode(self, observation):
		if self.encoder is None:
			return observation
		return self.encoder.encode(observation)


	@agnosticmethod
	def distance(self, observation1, observation2):
		return self.metric.distance(observation1, observation2)


	@agnosticmethod
	def true_distance(self, label1, label2):
		return self.dataset.space_of(self.target_key).distance(label1, label2, standardize=self.standardize_true)


	@agnosticmethod
	def compare(self, distances, true_distances):
		return self.criterion.compare(distances, true_distances)


	@agnosticmethod
	def _encode_step(self, info):
		info[self.latent_key] = self.encode(info[self.observation_key])
		return info


	@agnosticmethod
	def _generate_pairs(self, samples):
		if self.use_pairwise:
			i,j = torch.triu_indices(len(samples), len(samples), offset=1)
			return samples[i], samples[j]
		return samples.chunk(2)


	@agnosticmethod
	def _measure_distance(self, info):
		a, b = self._generate_pairs(info[self.latent_key])
		info['a'], info['b'] = a, b
		info[self.distance_key] = self.distance(a, b)
		return info


	@agnosticmethod
	def _true_distance(self, info):
		a, b = self._generate_pairs(info[self.target_key])
		info['label_a'], info['label_b'] = a, b
		info[self.true_distance_key] = self.true_distance(a,b)
		return info


	@agnosticmethod
	def _compare_distances(self, info):
		info[self.scores_key] = self.compare(info[self.distance_key].view(1,-1),
		                                     info[self.true_distance_key].view(1,-1))
		return info


	class ResultsContainer(AbstractMetricTask.ResultsContainer): # TODO: auto-accumulate scores_key
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
	def aggregate(self, info):
		info = super().aggregate(info)
		
		distances = info.aggregate(self.distance_key)
		true_distances = info.aggregate(self.true_distance_key)

		info.update({
			f'full_{self.distance_key}': distances,
			f'full_{self.true_distance_key}': true_distances,
		})
		return info


	# @agnosticmethod
	# def score_names(self):
	# 	return {*super().score_names()}


	@agnosticmethod
	def heavy_results(self):
		return {f'full_{self.distance_key}', f'full_{self.true_distance_key}', *super().heavy_results()}



class AdjustedCosineSimilarity(models.Criterion):
	def __init__(self, dim=1, eps=1e-8, lower_bound=0.75, **kwargs):
		super().__init__(**kwargs)
		self.lower_bound = lower_bound
		self.dim = dim
		self.eps = eps


	def compare(self, observation1, observation2):
		sim = F.cosine_similarity(observation1, observation2, dim=self.dim, eps=self.eps)
		if self.lower_bound is None:
			return sim
		return (sim - self.lower_bound) / (1 - self.lower_bound)



class SpearmanCorrelation(models.Criterion):
	def __init__(self, dim=1, full_mat=False, **kwargs):
		super().__init__(**kwargs)
		self.dim = dim
		self.full_mat = full_mat


	def compare(self, observation1, observation2): # TODO: use pytorch instead of numpy
		N1, N2 = len(observation1), len(observation2)
		rho, _ = stats.spearmanr(observation1.cpu().detach().numpy(), observation2.cpu().detach().numpy(), axis=1,
		                         nan_policy='raise')
		rho = torch.as_tensor(rho).to(observation1)
		if N1 == 1 and N2 == 1:
			return rho
		R = rho[N1:N1 + N2, :N1]
		if self.full_mat:
			return R
		return R.diag()



@inherit_hparams('encoder', 'dataset', 'use_pairwise', 'standardize_true')
class CorrelationMetricTask(MetricTask):
	measure_key = 'measure'
	true_measure_key = 'true_measure'

	correlation_element_key = 'batch_cross_correlation'
	modularity_element_key = 'batch_modularity'
	compactness_element_key = 'batch_compactness'

	correlation_mat_key = 'cross_correlation'

	modularity_key = 'modularity'
	compactness_key = 'compactness'
	# score_key = modularity_key

	metric_name = hparam('l2')
	keep_all_cor_mats = True

	@hparam(module=models.Criterion)
	def criterion(self):
		return self.correlator


	@hparam(cache=True, module=models.Criterion)
	def correlator(self):
		return SpearmanCorrelation(full_mat=True)


	@hparam(cache=True, module=models.Metric)
	def metric(self):
		return get_metric(self.metric_name)()


	@agnosticmethod
	def correlate(self, pred, true):
		return self.correlator.compare(pred.t(), true.t())


	@agnosticmethod
	def measure(self, observation1, observation2):
		return self.metric.measure(observation1, observation2)


	@agnosticmethod
	def true_measure(self, label1, label2):
		return self.dataset.space_of(self.target_key).measure(label1, label2)


	@agnosticmethod
	def _calc_modularity(self, mat):
		return math.mixing_score(mat, dim1=0, dim2=1)


	@agnosticmethod
	def _calc_compactness(self, mat):
		return math.mixing_score(mat, dim1=1, dim2=0)


	@agnosticmethod
	def _measure_distance(self, info):
		info = super()._measure_distance(info)
		info[self.measure_key] = self.measure(info['a'], info['b'])
		return info


	@agnosticmethod
	def _true_distance(self, info):
		info = super()._true_distance(info)
		info[self.true_measure_key] = self.true_measure(info['label_a'], info['label_b'])
		return info


	@agnosticmethod
	def _compare_distances(self, info):
		info = super()._compare_distances(info)
		info[self.correlation_element_key] = self.correlate(info[self.measure_key], info[self.true_measure_key])

		info[self.modularity_element_key] = self._calc_modularity(info[self.correlation_element_key]).item()
		info[self.compactness_element_key] = self._calc_compactness(info[self.correlation_element_key]).item()
		info[self.correlation_element_key] = info[self.correlation_element_key].unsqueeze(0)
		return info


	@agnosticmethod
	def aggregate(self, info):
		info = super().aggregate(info)

		cor_mats = info.aggregate(self.correlation_element_key)

		modularity = info.aggregate(self.modularity_element_key)
		compactness = info.aggregate(self.compactness_element_key)

		info.update({
			f'full_{self.correlation_element_key}': cor_mats,
			f'{self.correlation_mat_key}_mean': cor_mats.mean(0),
			f'{self.correlation_mat_key}_max': cor_mats.max(0)[0],
			f'{self.correlation_mat_key}_min': cor_mats.min(0)[0],
			f'{self.correlation_mat_key}_std': cor_mats.std(0),

			f'full_{self.modularity_element_key}': modularity,
			f'{self.modularity_key}_mean': modularity.mean().item(),
			f'{self.modularity_key}_max': modularity.max().item(),
			f'{self.modularity_key}_min': modularity.min().item(),
			f'{self.modularity_key}_std': modularity.std().item(),

			f'full_{self.compactness_element_key}': compactness,
			f'{self.compactness_key}_mean': compactness.mean().item(),
			f'{self.compactness_key}_max': compactness.max().item(),
			f'{self.compactness_key}_min': compactness.min().item(),
			f'{self.compactness_key}_std': compactness.std().item(),
		})

		if not self.keep_all_cor_mats or info[f'full_{self.correlation_element_key}'].numel() > 10000000:
			del info[f'full_{self.correlation_element_key}']

		info[self.correlation_mat_key] = info[f'{self.correlation_mat_key}_mean']
		info[self.modularity_key] = info[f'{self.modularity_key}_mean']
		info[self.compactness_key] = info[f'{self.compactness_key}_mean']
		return info


	@agnosticmethod
	def score_names(self):
		return {self.modularity_key, self.compactness_key,
		        f'{self.modularity_key}_max', f'{self.modularity_key}_min', f'{self.modularity_key}_std',
		        f'{self.compactness_key}_max', f'{self.compactness_key}_min', f'{self.compactness_key}_std',
		        *super().score_names()}


	@agnosticmethod
	def heavy_results(self):
		return {f'full_{self.compactness_element_key}', f'full_{self.modularity_element_key}',
		        f'full_{self.correlation_element_key}', *super().heavy_results()}


	class ResultsContainer(MetricTask.ResultsContainer): # TODO: auto-accumulate scores_key
		def __init__(self, correlation_element_key=None, modularity_element_key=None, compactness_element_key=None,
		             **kwargs):
			super().__init__(**kwargs)
			if correlation_element_key is not None and modularity_element_key is not None \
					and compactness_element_key is not None:
				self.register_cumulative(correlation_element_key, modularity_element_key, compactness_element_key)


	@agnosticmethod
	def create_results_container(self, info=None, correlation_element_key=None, modularity_element_key=None,
	                             compactness_element_key=None, **kwargs):
		if correlation_element_key is None:
			correlation_element_key = self.correlation_element_key
		if modularity_element_key is None:
			modularity_element_key = self.modularity_element_key
		if compactness_element_key is None:
			compactness_element_key = self.compactness_element_key
		return super().create_results_container(info=info, correlation_element_key=correlation_element_key,
		                                        modularity_element_key=modularity_element_key,
		                                        compactness_element_key=compactness_element_key, **kwargs)



@inherit_hparams('encoder', 'dataset', 'use_pairwise', 'standardize_true')
class CosSimMetricTask(MetricTask):

	metric_name = hparam('l2')


	lower_bound = hparam(0.75)
	@hparam(cache=True, module=models.Criterion)
	def criterion(self):
		return AdjustedCosineSimilarity(lower_bound=self.lower_bound)


	@hparam(cache=True, module=models.Metric)
	def metric(self):
		return get_metric(self.metric_name)()



