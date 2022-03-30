import torch
from omnibelt import get_printer, unspecified_argument
from ...framework import util
from ..base import Task, BatchedTask, SimpleEvaluationTask

prt = get_printer(__file__)


class AbstractGenerationTask(BatchedTask):
	@classmethod
	def run_step(cls, batch, info):
		info.clear()
		info.set_batch(batch)
		cls._generate_step(info)
		cls._judge_step(info)
		return info


	@staticmethod
	def _generate_step(info):
		raise NotImplementedError


	@staticmethod
	def _judge_step(info):
		raise NotImplementedError



class DiscriminatorGenerationTask(SimpleEvaluationTask, AbstractGenerationTask):
	def __init__(self, generator=None, discriminator=None, **kwargs):
		super().__init__(**kwargs)
		self.generator = generator
		self.discriminator = discriminator

	@classmethod
	def _eval_step(cls, info, *, comparison_key='comparison'):
		cls._generate_step(info)
		cls._judge_step(info, comparison_key=comparison_key)
		return info


	@staticmethod
	def _generate_step(info, generated_key='samples'):
		info[generated_key] = info.generator.generate(info.batch.size, gen=info.gen)
		return info
	

	@staticmethod
	def _judge_step(info, *, samples=None, comparison_key='comparison'):
		if samples is None:
			samples = info['samples']
		info[comparison_key] = info.discriminator.judge(samples)
		return info



class AbstractFeatureGenerationTask(AbstractGenerationTask):
	def __init__(self, generator=None, extractor=None, feature_criterion=None, **kwargs):
		super().__init__(**kwargs)
		self.generator = generator
		self.extractor = extractor
		self.feature_criterion = feature_criterion
	
	
	class ResultsContainer(AbstractGenerationTask.ResultsContainer):
		def __init__(self, **kwargs):
			super().__init__(**kwargs)
			self.fake_features = []
			self.real_features = []
		
		
		def accumulate_real(self, features):
			if self.real_features is not None:
				self.real_features.append(features)
		
		
		def accumulate_fake(self, features):
			self.fake_features.append(features)
		
		
		def aggregate_real(self):
			if self.real_features is not None:
				return torch.cat(self.real_features)
		
		
		def aggregate_fake(self):
			if self.fake_features is not None:
				return torch.cat(self.fake_features)
	
	
	def prepare(self, generator=None, extractor=None, **kwargs):
		if generator is None:
			prt.warning('No generator provided')
		if extractor is None:
			prt.warning('No extractor provided')
		info = super().prepare(**kwargs)
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






