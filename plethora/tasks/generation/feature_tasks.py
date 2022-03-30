from omnibelt import unspecified_argument
import torch
from torch import nn
import piq
from piq import IS, FID, GS, KID, MSID, PR
from piq import inception_score
from piq.fid import _compute_statistics as compute_fid_stats, _compute_fid as compute_frechet_distance

from .task import AbstractFeatureGenerationTask



class IS_GenerationTask(AbstractFeatureGenerationTask):
	'''
	"A Note on the Inception Score" https://arxiv.org/pdf/1801.01973.pdf
	'''
	def __init__(self, num_splits=10, metric='l1', **kwargs):
		super().__init__(**kwargs)
		self.num_splits = num_splits
		self.metric = self._find_metric(metric)
	
	
	score_key = 'inception_score'
	
	
	@staticmethod
	def _find_metric(ident, p=2, **kwargs):
		if not isinstance(ident, str):
			return ident
		
		if ident.lower() == 'l1':
			return nn.PairwiseDistance(p=1, **kwargs)
		elif ident.lower() == 'l2':
			return nn.PairwiseDistance(p=2, **kwargs)
		elif ident.lower() == 'lp':
			return nn.PairwiseDistance(p=p, **kwargs)
		else:
			raise NotImplementedError(ident)
	
	
	def prepare(self, **kwargs):
		info = super().prepare(**kwargs)
		info.num_splits = self.num_splits
		info.metric = self.metric
	
		
	@staticmethod
	def inception_score(features, num_splits=10):
		mu, sigma = inception_score(features, num_splits=num_splits)
		return mu, sigma
	
	
	@classmethod
	def _feature_criterion_step(cls, info, fake, real):
		
		fake_mu, fake_std = cls.inception_score(fake, num_splits=info.num_splits)
		real_mu, real_std = cls.inception_score(real, num_splits=info.num_splits)
		
		info.update({
			'fake_mu': fake_mu, 'fake_std': fake_std,
			'real_mu': real_mu, 'real_std': real_std,
		})
		score = info.metric(fake_mu.view(1, -1), real_mu.view(1, -1)).item()
		info['inception_score'] = score
		return info
	


class FID_GenerationTask(AbstractFeatureGenerationTask):
	score_key = 'fid'
	
	@staticmethod
	def compute_fid_stats(features):
		return compute_fid_stats(features)
	
	
	@staticmethod
	def compute_frechet_distance(mu1, sigma1, mu2, sigma2):
		return compute_frechet_distance(mu1, sigma1, mu2, sigma2)
	
	
	@classmethod
	def _feature_criterion_step(cls, info, fake, real):
		fake_mu, fake_std = cls.compute_fid_stats(fake)
		real_mu, real_std = cls.compute_fid_stats(real)
		
		info.update({
			'fake_mu': fake_mu, 'fake_std': fake_std,
			'real_mu': real_mu, 'real_std': real_std,
		})
		score = cls.compute_frechet_distance(fake_mu, fake_std, real_mu, real_std).item()
		info['fid'] = score
		return info



class GS_GenerationTask(AbstractFeatureGenerationTask):
	def __init__(self, sample_size=64, num_iters=1000, gamma=None, i_max=100, num_workers=4, **kwargs):
		super().__init__(**kwargs)
		self.sample_size = sample_size
		self.num_iters = num_iters
		self.gamma = gamma
		self.i_max = i_max
		self.num_workers = num_workers
		
		
	score_key = 'geometry_score'


	def prepare(self, **kwargs):
		info = super().prepare(**kwargs)
		info.feature_criterion = GS(sample_size=self.sample_size, num_iters=self.num_iters, gamma=self.gamma,
		                            i_max=self.i_max, num_workers=self.num_workers)
		return info
	
	
	@staticmethod
	def _feature_criterion_step(info, fake, real):
		info['geometry_score'] = info.feature_criterion.compute_metric(fake, real)
		return info



class KID_GenerationTask(AbstractFeatureGenerationTask):
	def __init__(self, degree=3, gamma=None, coef0=1, var_at_m=None, average=False,
	             n_subsets=50, subset_size=1000, **kwargs):
		super().__init__(**kwargs)
		self.degree = degree
		self.gamma = gamma
		self.coef0 = coef0
		self.var_at_m = var_at_m
		self.average = average
		self.n_subsets = n_subsets
		self.subset_size = subset_size

	
	score_key = 'kid'
	

	def prepare(self, **kwargs):
		info = super().prepare(**kwargs)
		info.feature_criterion = KID(degree=self.degree, gamma=self.gamma, coef0=self.coef0, var_at_m=self.var_at_m,
		                             average=self.average, n_subsets=self.n_subsets, subset_size=self.subset_size,
		                             ret_var=True)
		return info


	@staticmethod
	def _feature_criterion_step(info, fake, real):
		info['kid'], info['kid_variance'] = info.feature_criterion.compute_metric(fake, real)
		return info
	


class MSID_GenerationTask(AbstractFeatureGenerationTask):
	def __init__(self, ts=None, k=5, m=10, niters=100, rademacher=False, normalized_laplacian=True,
	             normalize='empty', msid_mode='max', **kwargs):
		super().__init__(**kwargs)
		self.ts = ts
		self.k = k
		self.m = m
		self.niters = niters
		self.rademacher = rademacher
		self.normalized_laplacian = normalized_laplacian
		self.normalize = normalize
		_mode_options = {'l2', 'max'}
		if msid_mode not in _mode_options:
			raise ValueError(f'msid_mode must be in {_mode_options}')
		self.msid_mode = msid_mode
	
	
	score_key = 'msid'
	
	
	def prepare(self, **kwargs):
		info = super().prepare(**kwargs)
		info.feature_criterion = MSID(ts=self.ts, k=self.k, m=self.m, niters=self.niters, rademacher=self.rademacher,
		                              normalized_laplacian=self.normalized_laplacian, normalize=self.normalize,
		                              msid_mode=self.msid_mode)
		return info
	
	
	@staticmethod
	def _feature_criterion_step(info, fake, real):
		info['msid'] = info.feature_criterion.compute_metric(fake, real)
		return info



class PR_GenerationTask(AbstractFeatureGenerationTask):
	def __init__(self, nearest_k=5, **kwargs):
		super().__init__(**kwargs)
		self.nearest_k = nearest_k


	score_key =  'f1'


	def prepare(self, **kwargs):
		info = super().prepare(**kwargs)
		info.feature_criterion = PR(nearest_k=self.nearest_k)
		return info


	@staticmethod
	def _feature_criterion_step(info, fake, real):
		precision, recall = info.feature_criterion.compute_metric(real, fake)
		info['precision'], info['recall'] = precision, recall
		info['f1'] = 2 * precision * recall / (precision + recall)
		return info



