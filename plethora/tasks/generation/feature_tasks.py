from omnibelt import unspecified_argument, agnosticmethod
import torch
from torch import nn
import piq
from piq import IS, FID, GS, KID, MSID, PR
from piq import inception_score
from piq.fid import _compute_statistics as compute_fid_stats, _compute_fid as compute_frechet_distance

from ...framework import with_modules, with_hparams
from .task import AbstractFeatureGenerationTask


@with_hparams(num_splits=10, metric='l1')
class IS_GenerationTask(AbstractFeatureGenerationTask):
	'''
	"A Note on the Inception Score" https://arxiv.org/pdf/1801.01973.pdf
	'''

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


	@staticmethod
	def inception_score(features, num_splits=10):
		mu, sigma = inception_score(features, num_splits=num_splits)
		return mu, sigma
	
	
	@agnosticmethod
	def _compare_features(self, info, *, fake=None, real=None):
		if fake is None:
			fake = info[self.fake_features_key]
		if real is None:
			real = info[self.real_features_key]
		
		fake_mu, fake_std = self.inception_score(fake, num_splits=info.num_splits)
		real_mu, real_std = self.inception_score(real, num_splits=info.num_splits)
		
		info.update({
			'fake_mu': fake_mu, 'fake_std': fake_std,
			'real_mu': real_mu, 'real_std': real_std,
		})

		metric = self._find_metric(info.metric)
		score = metric(fake_mu.view(1, -1), real_mu.view(1, -1)).item()
		info[self.score_key] = score
		return info
	


class FID_GenerationTask(AbstractFeatureGenerationTask):
	score_key = 'fid'
	
	@staticmethod
	def compute_fid_stats(features):
		return compute_fid_stats(features)
	
	
	@staticmethod
	def compute_frechet_distance(mu1, sigma1, mu2, sigma2):
		return compute_frechet_distance(mu1, sigma1, mu2, sigma2)
	
	
	@agnosticmethod
	def _compare_features(self, info, *, fake=None, real=None):
		if fake is None:
			fake = info[self.fake_features_key]
		if real is None:
			real = info[self.real_features_key]

		fake_mu, fake_std = self.compute_fid_stats(fake)
		real_mu, real_std = self.compute_fid_stats(real)
		
		info.update({
			'fake_mu': fake_mu, 'fake_std': fake_std,
			'real_mu': real_mu, 'real_std': real_std,
		})
		score = self.compute_frechet_distance(fake_mu, fake_std, real_mu, real_std).item()
		info[self.score_key] = score
		return info



class FIDInfinity_GenerationTask(FID_GenerationTask):
	@agnosticmethod
	def _compare_features(self, info, *, fake=None, real=None): # TODO
		raise NotImplementedError



@with_hparams(sample_size=64, num_iters=1000, gamma=None, i_max=100, num_workers=4)
class GS_GenerationTask(AbstractFeatureGenerationTask):
	score_key = 'geometry_score'

	@agnosticmethod
	def _compare_features(self, info, *, fake=None, real=None):
		if fake is None:
			fake = info[self.fake_features_key]
		if real is None:
			real = info[self.real_features_key]

		criterion = GS(sample_size=info.sample_size, num_iters=info.num_iters, gamma=info.gamma, i_max=info.i_max,
		               num_workers=info.num_workers)
		info[self.score_key] = criterion.compute_metric(fake, real)
		return info



@with_hparams(degree=3, gamma=None, coef0=1, var_at_m=None, average=False, n_subsets=50, subset_size=1000)
class KID_GenerationTask(AbstractFeatureGenerationTask):
	score_key = 'kid'


	@agnosticmethod
	def _compare_features(self, info, *, fake=None, real=None):
		if fake is None:
			fake = info[self.fake_features_key]
		if real is None:
			real = info[self.real_features_key]

		criterion = KID(degree=info.degree, gamma=info.gamma, coef0=info.coef0, var_at_m=info.var_at_m,
                         average=info.average, n_subsets=info.n_subsets, subset_size=info.subset_size,
                         ret_var=True)

		info[self.score_key], info['kid_variance'] = criterion.compute_metric(fake, real)
		return info



@with_hparams(ts=None, k=5, m=10, niters=100, rademacher=False, normalized_laplacian=True,
              normalize='empty', msid_mode='max')
class MSID_GenerationTask(AbstractFeatureGenerationTask):
	score_key = 'msid'
	

	@agnosticmethod
	def _compare_features(self, info, *, fake=None, real=None):
		if fake is None:
			fake = info[self.fake_features_key]
		if real is None:
			real = info[self.real_features_key]

		criterion = MSID(ts=info.ts, k=info.k, m=info.m, niters=info.niters, rademacher=info.rademacher,
		                 normalized_laplacian=info.normalized_laplacian, normalize=info.normalize,
		                 msid_mode=info.msid_mode)
		info[self.score_key] = criterion.compute_metric(fake, real)
		return info



@with_hparams(nearest_k=5)
class PR_GenerationTask(AbstractFeatureGenerationTask):
	score_key =  'f1'


	@agnosticmethod
	def _compare_features(self, info, *, fake=None, real=None):
		if fake is None:
			fake = info[self.fake_features_key]
		if real is None:
			real = info[self.real_features_key]

		criterion = PR(nearest_k=info.nearest_k)
		precision, recall = criterion.compute_metric(real, fake)
		info['precision'], info['recall'] = precision, recall
		info[self.score_key] = 2 * precision * recall / (precision + recall)
		return info



