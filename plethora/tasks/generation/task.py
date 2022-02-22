import torch
from omnibelt import get_printer
from ...framework import util
from ..base import Task, BatchedTask, ResultsContainer

prt = get_printer(__file__)


class AccumulationContainer(ResultsContainer):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.criteria = []


	def accumulate(self, criteria):
		self.criteria.append(criteria)


	def aggregate(self):
		return torch.cat(self.criteria)



class AbstractGenerationTask(Task):
	@classmethod
	def run(cls, info, **kwargs):
		
		
		
		info.clear()
		info.set_batch(batch)
		cls._generate(info)
		cls._eval_samples(info)
		return info


	@staticmethod
	def _generate_batch(info):
		raise NotImplementedError


	@staticmethod
	def _eval_samples(info):
		raise NotImplementedError



class DatasetGenerationTask(BatchedTask):
	
	@staticmethod
	def run_step(batch, info, slim=False, online=False, seed=None, gen=None):
		
		pass
	
	pass



class GenerationTask(AbstractGenerationTask):
	def __init__(self, generator=None, discriminator=None,
	             score_key=None, **kwargs):
		if score_key is None:
			score_key = 'mean'
		super().__init__(score_key=score_key, **kwargs)

		self.generator = generator
		self.discriminator = discriminator


	@staticmethod
	def score_names():
		return ['mean', 'std', 'min', 'max', *super().score_names()]

	@staticmethod
	def create_results_container(dataset=None, **kwargs):
		return AccumulationContainer(dataset=dataset, **kwargs)

	def _compute(self, **kwargs):
		return super()._compute(generator=self.generator, discriminator=self.discriminator, **kwargs)

	@classmethod
	def prepare(cls, generator=None, discriminator=None, **kwargs):
		if generator is None:
			prt.warning('No generator provided')
		if discriminator is None:
			prt.warning('No discriminator provided')
		info = super().prepare(**kwargs)
		info.generator = generator
		info.discriminator = discriminator
		return info


	@staticmethod
	def _generate(info):
		# info.generator.generate(, gen=info.gen)
		raise NotImplementedError


	@staticmethod
	def _eval_samples(info):
		raise NotImplementedError


	@classmethod
	def aggregate(cls, info, slim=False, online=False, seed=None, gen=None):
		info = super().aggregate(info)

		criteria = info.aggregate()
		path_criteria = criteria.view(criteria.shape[0], -1).mean(-1)

		info.update({
			'criteria': criteria,
			'mean': path_criteria.mean(),
			'max': path_criteria.max(),
			'min': path_criteria.min(),
			'std': path_criteria.std(),
		})
		return info







