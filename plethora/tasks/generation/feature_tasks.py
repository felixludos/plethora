from omnibelt import unspecified_argument, agnosticmethod
import torch
from torch import nn
import piq
from piq import IS, FID, GS, KID, MSID, PR
from piq import inception_score
from piq.fid import _compute_statistics as compute_fid_stats, _compute_fid as compute_frechet_distance
from ...framework import models, hparam, inherit_hparams, data_args
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
	

	@data_args(score_key, fake=FeatureGenerationTask.fake_features_key, real=FeatureGenerationTask.real_features_key)
	def _compare_features(self, info, fake, real):
		fake_mu, fake_std = self.inception_score(fake)
		real_mu, real_std = self.inception_score(real)
		
		info.update({
			'fake_mu': fake_mu, 'fake_std': fake_std,
			'real_mu': real_mu, 'real_std': real_std,
		})

		return self.metric(fake_mu.view(1, -1), real_mu.view(1, -1)).item()
	


@inherit_hparams('generator', 'extractor')
class FID_GenerationTask(FeatureGenerationTask):
	score_key = 'fid'
	
	@staticmethod
	def compute_fid_stats(features):
		return compute_fid_stats(features)
	
	
	@staticmethod
	def compute_frechet_distance(mu1, sigma1, mu2, sigma2):
		return compute_frechet_distance(mu1, sigma1, mu2, sigma2)
	

	@data_args(score_key, fake=FeatureGenerationTask.fake_features_key, real=FeatureGenerationTask.real_features_key)
	def _compare_features(self, info, fake, real):
		fake_mu, fake_std = self.compute_fid_stats(fake)
		real_mu, real_std = self.compute_fid_stats(real)

		info.update({
			'fake_mu': fake_mu, 'fake_std': fake_std,
			'real_mu': real_mu, 'real_std': real_std,
		})
		return self.compute_frechet_distance(fake_mu, fake_std, real_mu, real_std).item()



@inherit_hparams('generator', 'extractor')
class FIDInfinity_GenerationTask(FID_GenerationTask):
	@agnosticmethod
	def _compare_features(self, info, fake, real): # TODO
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


	@data_args(score_key, fake=FeatureGenerationTask.fake_features_key, real=FeatureGenerationTask.real_features_key)
	def _compare_features(self, info):
		fake, real = self._extract_info(info, self.fake_features_key, self.real_features_key)
		return super()._compare_features(fake, real)



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


	@data_args(score_key)
	def _compare_features(self, info):
		score, variance = super()._compare_features(info, _process_out=False)
		info['kid_variance'] = variance
		return score



@inherit_hparams('generator', 'extractor')
class MSID_GenerationTask(FeatureGenerationTask):
	score_key = 'msid'

	ts = hparam(None)
	k = hparam(5)
	m = hparam(10)
	niters = hparam(100)
	rademacher = hparam(False)
	normalized_laplacian = hparam(True)
	normalize = hparam('empty')
	msid_mode = hparam('max')


	@hparam(cache=True)
	def feature_criterion(self):
		return MSID(ts=self.ts, k=self.k, m=self.m, niters=self.niters, rademacher=self.rademacher,
		                 normalized_laplacian=self.normalized_laplacian, normalize=self.normalize,
		                 msid_mode=self.msid_mode)


	@data_args(score_key, fake=FeatureGenerationTask.fake_features_key, real=FeatureGenerationTask.real_features_key)
	def _compare_features(self, fake, real):
		return super()._compare_features(fake, real)




@inherit_hparams('generator', 'extractor')
class PR_GenerationTask(FeatureGenerationTask):
	score_key =  'f1'

	nearest_k = hparam(5)


	@hparam(cache=True)
	def feature_criterion(self):
		return PR(nearest_k=self.nearest_k)


	@data_args(score_key, fake=FeatureGenerationTask.fake_features_key, real=FeatureGenerationTask.real_features_key)
	def _compare_features(self, info, fake, real):
		precision, recall = super()._compare_features(fake, real)
		info['precision'], info['recall'] = precision, recall
		return 2 * precision * recall / (precision + recall)


	# @data_args(score_key, fake='fake', real='real')
	# def _compare_features(self, info, fake, real):
	# 	precision, recall = self.feature_criterion.compute_metric(real, fake)
	# 	info['precision'], info['recall'] = precision, recall
	# 	return 2 * precision * recall / (precision + recall)



