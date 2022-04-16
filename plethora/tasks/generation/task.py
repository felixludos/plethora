import torch
from omnibelt import get_printer, unspecified_argument, agnosticmethod
from ...framework import models, hparam, inherit_hparams, data_args
from ..base import Task, BatchedTask, SimpleEvaluationTask, Cumulative

prt = get_printer(__file__)



class AbstractGenerationTask(BatchedTask):
	generator = hparam(module=models.Generator)


	@agnosticmethod
	def run_step(self, info):
		self._generate_step(info=info)
		self._judge_step(info=info)
		return info


	@data_args('generated')
	def _generate_step(self, info):
		return self.generator.generate(info.batch.size, gen=info.gen)

	# @agnosticmethod
	# def _generate_step(self, info):
	# 	info['generated'] = self.generator.generate(info.batch.size, gen=info.gen)
	# 	return info


	@agnosticmethod
	def _judge_step(self, info):
		raise NotImplementedError



@inherit_hparams('generator')
class DiscriminatorGenerationTask(SimpleEvaluationTask, AbstractGenerationTask):
	discriminator = hparam(module=models.Discriminator)

	@data_args('scores', samples='generated')
	def _judge_step(self, samples):
		return self.discriminator.judge(samples)

	# @agnosticmethod
	# def _judge_step(self, info):
	# 	info['scores'] = self.discriminator.judge(info['generated'])
	# 	return info


@inherit_hparams('generator')
class FeatureGenerationTask(AbstractGenerationTask):
	fake_key = 'fake'
	real_key = 'real'
	fake_features_key = 'fake_features'
	real_features_key = 'real_features'


	feature_criterion = hparam()
	extractor = hparam(module=models.Extractor)


	@data_args(observation='observation', generated='generated')
	def _judge_step(self, info, observation, generated):
		info[self.fake_key] = generated if self.extractor is None else self.extractor.encode(generated)
		info[self.real_key] = observation if self.extractor is None else self.extractor.encode(observation)
		return info


	@agnosticmethod
	def heavy_results(self):
		return {self.fake_features_key, self.real_features_key, *super().heavy_results()}


	@agnosticmethod
	def aggregate(self, info):
		info = super().aggregate(info)
		info[self.fake_features_key] = info.aggregate(self.fake_key)
		info[self.real_features_key] = info.aggregate(self.real_key)
		self._compare_features(info)
		return info


	@agnosticmethod
	def compare_features(self, fake, real):
		return self.feature_criterion.compute_metric(fake, real)


	@agnosticmethod
	def _compare_features(self, info):
		info[self.score_key] = self.compare_features(info[self.fake_features_key], info[self.real_features_key])
		return info


	# @data_args('score', fake=fake_features_key, real=real_features_key)
	# def _compare_features(self, fake, real):
	# 	return self.feature_criterion.compute_metric(fake, real)







