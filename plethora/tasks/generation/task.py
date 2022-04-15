import torch
from omnibelt import get_printer, unspecified_argument, agnosticmethod
from ...framework import util#, with_hparams, with_modules, models, with_args
from ..base import Task, BatchedTask, SimpleEvaluationTask, Cumulative

prt = get_printer(__file__)


class auto_args:
	def __init__(self, _out_key=None, **args):
		self._out_key = None
		self.args = args


	def __call__(self, *args, **kwargs):
		
		pass


	# def __get__(self, instance, owner):
	# 	pass


@with_modules(generator=models.Generator)
class AbstractGenerationTask(BatchedTask):
	generated_key = 'generated'

	@agnosticmethod
	def run_step(self, info):
		info.clear()
		self._generate_step(info)
		self._judge_step(info)
		return info


	@auto_args('eval', observation='observation')
	@auto_attrs('eval', observation='observation')
	def judge(self, observation=None):


	@agnosticmethod
	def _generate_step(self, info):
		info[self.generated_key] = info.generator.generate(info.batch.size, gen=info.gen)
		return info


	@staticmethod
	def _judge_step(info):
		raise NotImplementedError



@with_modules(discriminator=models.Discriminator)
class DiscriminatorGenerationTask(SimpleEvaluationTask, AbstractGenerationTask):

	@agnosticmethod
	def _judge_step(self, info, *, generated=None):
		if generated is None:
			generated = info[self.generated_key]
		info[info.evaluation_key] = info.discriminator.judge(generated)
		return info



@with_modules(extractor=models.Extractor, required=False)
class AbstractFeatureGenerationTask(AbstractGenerationTask):
	observation_key = 'observation'

	fake_key = 'fake'
	real_key = 'real'
	fake_features_key = 'fake_features'
	real_features_key = 'real_features'


	@agnosticmethod
	def _judge_step(self, info, *, observation=None, generated=None):
		if generated is None:
			generated = info[self.generated_key]
		if observation is None:
			observation = info[self.observation_key]

		info[self.fake_key] = generated if info.extractor is None else info.extractor.encode(generated)
		info[self.real_key] = observation if info.extractor is None else info.extractor.encode(observation)
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


	@staticmethod
	def _compare_features(info, *, fake=None, real=None):
		raise NotImplementedError






