import torch
from omnibelt import get_printer, unspecified_argument, agnosticmethod
from ...framework import util
from ..base import Task, BatchedTask, SimpleEvaluationTask, Cumulative

prt = get_printer(__file__)


class AbstractGenerationTask(BatchedTask):
	@classmethod
	def run_step(cls, info):
		info.clear()
		cls._generate_step(info)
		cls._judge_step(info)
		return info


	@staticmethod
	def _generate_step(info, generated_key='generated'):
		info[generated_key] = info.generator.generate(info.batch.size, gen=info.gen)
		return info


	@staticmethod
	def _judge_step(info):
		raise NotImplementedError



class DiscriminatorGenerationTask(SimpleEvaluationTask, AbstractGenerationTask):
	def __init__(self, generator=None, discriminator=None, **kwargs):
		super().__init__(**kwargs)
		self.generator = generator
		self.discriminator = discriminator


	@staticmethod
	def _judge_step(info, *, generated=None):
		if generated is None:
			generated = info['generated']
		info[info.evaluation_key] = info.discriminator.judge(generated)
		return info



class AbstractFeatureGenerationTask(AbstractGenerationTask):
	def __init__(self, generator=None, extractor=None, **kwargs):
		super().__init__(**kwargs)
		self.generator = generator
		self.extractor = extractor
	
	
	class ResultsContainer(Cumulative, AbstractGenerationTask.ResultsContainer):
		_auto_cumulative_keys = ['real', 'fake']


	def create_results_container(self, generator=None, extractor=None, **kwargs):
		if generator is None:
			generator = self.generator
		if extractor is None:
			extractor = self.extractor
		info = super().create_results_container(**kwargs)
		info.generator = generator
		info.extractor = extractor
		return info
	
	
	@staticmethod
	def _generate_step(info, *, real=None, generated_key='generated'):
		info[generated_key] = info.generator.generate(info.batch.size, gen=info.gen)
		return info
	
	
	@staticmethod
	def _judge_step(info, *, observation=None, generated=None, real_key='real', fake_key='fake'):
		if observation is None:
			observation = info['observation']
		if generated is None:
			generated = info['generated']

		info[fake_key] = generated if info.extractor is None else info.extractor.encode(generated)
		info[real_key] = observation if info.extractor is None else info.extractor.encode(observation)

		info.accumulate_fake(info[fake_key])
		info.accumulate_real(info[real_key])
		return info


	@classmethod
	def aggregate(cls, info):
		info = super().aggregate(info)
		
		info['fake_features'] = info.aggregate_fake()
		info['real_features'] = info.aggregate_real()
		
		cls._feature_criterion_step(info, info['fake_features'], info['real_features'])
		
		if info.slim:
			del info['fake_features']
			del info['real_features']
		return info


	@staticmethod
	def _feature_criterion_step(info, fake, real):
		raise NotImplementedError






