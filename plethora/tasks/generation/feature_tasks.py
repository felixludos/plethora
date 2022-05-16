from omnibelt import unspecified_argument, agnosticmethod
import torch
from torch import nn
import piq
from piq import IS, FID, GS, KID, MSID, PR
from piq import inception_score
from piq.fid import _compute_statistics as compute_fid_stats, _compute_fid as compute_frechet_distance
from ...framework import models, hparam, inherit_hparams
from .task import FeatureGenerationTask


@inherit_hparams('generator', 'extractor')
class IS_GenerationTask(FeatureGenerationTask):
	'''
	"A Note on the Inception Score" https://arxiv.org/pdf/1801.01973.pdf
	'''

	score_key = 'inception_score'

	num_splits = hparam(10)
	metric_name = hparam('l1', space=['l1', 'l2'])


	@hparam(module=models.Metric)
	def metric(self):
		ident = self.metric_name
		if ident.lower() == 'l1':
			return nn.PairwiseDistance(p=1)
		elif ident.lower() == 'l2':
			return nn.PairwiseDistance(p=2)
		else:
			raise NotImplementedError(ident)


	@agnosticmethod
	def inception_score(self, features):
		mu, sigma = inception_score(features, num_splits=self.num_splits)
		return mu, sigma
	

	@agnosticmethod
	def _compare_features(self, info):
		fake_mu, fake_std = self.inception_score(info[self.fake_features_key])
		real_mu, real_std = self.inception_score(info[self.real_features_key])
		
		info.update({
			'fake_inception_mu': fake_mu, 'fake_inception_std': fake_std,
			'real_inception_mu': real_mu, 'real_inception_std': real_std,
		})

		info[self.score_key] = self.metric(fake_mu.view(1, -1), real_mu.view(1, -1)).item()
		return info



@inherit_hparams('generator', 'extractor')
class FID_GenerationTask(FeatureGenerationTask):
	score_key = 'fid'
	
	_eps = 1e-6
	
	@staticmethod
	def compute_fid_stats(features):
		return compute_fid_stats(features)
	
	
	@staticmethod
	def compute_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
		return compute_frechet_distance(mu1, sigma1, mu2, sigma2, eps=eps)
	
	
	@agnosticmethod
	def _compare_features(self, info):
		fake_mu, fake_cov = self.compute_fid_stats(info[self.fake_features_key])
		real_mu, real_cov = self.compute_fid_stats(info[self.real_features_key])

		info.update({
			'fake_fid_mu': fake_mu, 'fake_fid_cov': fake_cov,
			'real_fid_mu': real_mu, 'real_fid_cov': real_cov,
		})
		score = self.compute_frechet_distance(fake_mu, fake_cov, real_mu, real_cov, eps=self._eps).item()
		info[self.score_key] = score
		return info



@inherit_hparams('generator', 'extractor')
class FIDInfinity_GenerationTask(FID_GenerationTask):
	@agnosticmethod
	def _compare_features(self, info): # TODO
		raise NotImplementedError



@inherit_hparams('generator', 'extractor')
class GS_GenerationTask(FeatureGenerationTask):
	score_key = 'geometry_score'

	sample_size = hparam(64)
	num_iters = hparam(1000)
	gamma = hparam(None)
	i_max = hparam(100)
	num_workers = hparam(4)


	@hparam(cache=True)
	def feature_criterion(self):
		return GS(sample_size=self.sample_size, num_iters=self.num_iters, gamma=self.gamma, i_max=self.i_max,
		               num_workers=self.num_workers)



@inherit_hparams('generator', 'extractor')
class KID_GenerationTask(FeatureGenerationTask):
	score_key = 'kid'

	degree = hparam(3)
	gamma = hparam(None)
	coef0 = hparam(1)
	var_at_m = hparam(None)
	average = hparam(False)
	n_subsets = hparam(50)
	subset_size = hparam(1000)


	@hparam(cache=True)
	def feature_criterion(self):
		return KID(degree=self.degree, gamma=self.gamma, coef0=self.coef0, var_at_m=self.var_at_m,
                         average=self.average, n_subsets=self.n_subsets, subset_size=self.subset_size,
                         ret_var=True)

	
	@agnosticmethod
	def _compare_features(self, info):
		info = super()._compare_features(info)
		info[self.score_key], info['kid_var'] = info[self.score_key]
		return info


	@agnosticmethod
	def score_names(self):
		return {'kid_var', *super().score_names()}



@inherit_hparams('generator', 'extractor')
class MSID_GenerationTask(FeatureGenerationTask):
	score_key = 'msid'

	ts = hparam(None)
	k = hparam(5)
	m = hparam(10)
	niters = hparam(100)
	rademacher = hparam(False)
	normalized_laplacian = hparam(True)
	normalize = hparam('empty', space=['empty', 'complete', 'er'])
	msid_mode = hparam('max', space=['max', 'l2'])


	@hparam(cache=True)
	def feature_criterion(self):
		return MSID(ts=self.ts, k=self.k, m=self.m, niters=self.niters, rademacher=self.rademacher,
		                 normalized_laplacian=self.normalized_laplacian, normalize=self.normalize,
		                 msid_mode=self.msid_mode)



@inherit_hparams('generator', 'extractor')
class PR_GenerationTask(FeatureGenerationTask):
	score_key =  'f1'

	nearest_k = hparam(5)


	@hparam(cache=True)
	def feature_criterion(self):
		return PR(nearest_k=self.nearest_k)


	@agnosticmethod
	def score_names(self):
		return {'precision', 'recall', *super().score_names()}


	@agnosticmethod
	def _compare_features(self, info):
		info = super()._compare_features(info)
		precision, recall = info[self.score_key]
		precision, recall = precision.item(), recall.item()
		info['precision'], info['recall'] = precision, recall
		info[self.score_key] = 2 * precision * recall / (precision + recall)
		return info





