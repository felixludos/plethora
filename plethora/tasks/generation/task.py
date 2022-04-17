import torch
from omnibelt import get_printer, unspecified_argument, agnosticmethod
from ...framework import models, hparam, inherit_hparams
from ..base import Task, BatchedTask, SimpleEvaluationTask, Cumulative

prt = get_printer(__file__)



class AbstractGenerationTask(BatchedTask):
	generated_key = 'generated'
	
	generator = hparam(module=models.Generator)


	@agnosticmethod
	def run_step(self, info):
		self._generate_step(info=info)
		self._judge_step(info=info)
		return info


	@agnosticmethod
	def _generate_step(self, info):
		info[self.generated_key] = self.generator.generate(info.batch.size, gen=info.gen)
		return info


	@agnosticmethod
	def _judge_step(self, info):
		raise NotImplementedError



@inherit_hparams('generator')
class DiscriminatorGenerationTask(SimpleEvaluationTask, AbstractGenerationTask):
	discriminator = hparam(module=models.Discriminator)


	@agnosticmethod
	def _judge_step(self, info):
		info[self.scores_key] = self.judge(info[self.generated_key])
		return info


	@agnosticmethod
	def judge(self, samples):
		return self.discriminator.judge(samples)



@inherit_hparams('generator')
class FeatureGenerationTask(AbstractGenerationTask):
	observation_key = 'observation'
	
	fake_key = 'fake'
	real_key = 'real'
	fake_features_key = 'fake_features'
	real_features_key = 'real_features'


	feature_criterion = hparam()
	extractor = hparam(module=models.Extractor)


	@agnosticmethod
	def _judge_step(self, info):
		info[self.fake_key] = self.extract(info[self.generated_key])
		info[self.real_key] = self.extract(info[self.observation_key])
		return info

	
	@agnosticmethod
	def extract(self, observation):
		return observation if self.extractor is None else self.extractor.extract(observation)


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







