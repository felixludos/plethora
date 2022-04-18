import torch
from omnibelt import get_printer, unspecified_argument, agnosticmethod
from ...framework import models, hparam, inherit_hparams
from ..base import Task, BatchedTask, SimpleEvaluationTask, Cumulative

prt = get_printer(__file__)



class AbstractGenerationTask(BatchedTask):
	generated_key = 'generated'
	
	generator = hparam(module=models.Generator)


	@agnosticmethod
	def _compute_step(self, info):
		self._generate_step(info=info)
		self._judge_step(info=info)
		return info


	@agnosticmethod
	def generate(self, N, gen=None):
		return self.generator.generate(N, gen=gen)


	@agnosticmethod
	def _generate_step(self, info):
		info[self.generated_key] = self.generate(info.batch.size, gen=info.gen)
		return info


	@agnosticmethod
	def _judge_step(self, info):
		raise NotImplementedError



@inherit_hparams('generator')
class DiscriminatorGenerationTask(SimpleEvaluationTask, AbstractGenerationTask):
	discriminator = hparam(module=models.Discriminator)

	@agnosticmethod
	def judge(self, samples):
		return self.discriminator.judge(samples)


	@agnosticmethod
	def _judge_step(self, info):
		info[self.scores_key] = self.judge(info[self.generated_key])
		return info



@inherit_hparams('generator')
class FeatureGenerationTask(AbstractGenerationTask):
	observation_key = 'observation'
	
	fake_key = 'fake'
	real_key = 'real'
	fake_features_key = 'fake_features'
	real_features_key = 'real_features'


	feature_criterion = hparam()
	extractor = hparam(None, module=models.Extractor)


	@agnosticmethod
	def _judge_step(self, info):
		info[self.fake_key] = self.extract(info[self.generated_key])
		info[self.real_key] = self.extract(info[self.observation_key])
		return info


	@agnosticmethod
	def extract(self, observation): # TODO: maybe automatically reshape when theres no extractor (B, -1)
		if self.extractor is None:
			return observation.view(observation.shape[0], -1)
		return self.extractor(observation)


	class ResultsContainer(Cumulative, BatchedTask.ResultsContainer): # TODO: auto-accumulate scores_key
		def __init__(self, fake_key=None, real_key=None, **kwargs):
			super().__init__(**kwargs)
			if fake_key is not None and real_key is not None:
				self.register_cumulative(fake_key, real_key)


	@agnosticmethod
	def create_results_container(self, info=None, real_key=None, fake_key=None, **kwargs):
		if real_key is None:
			real_key = self.real_key
		if fake_key is None:
			fake_key = self.fake_key
		return super().create_results_container(info=info, real_key=real_key, fake_key=fake_key, **kwargs)


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







