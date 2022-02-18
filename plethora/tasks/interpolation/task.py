
import torch
from omnibelt import get_printer
from ...datasets.base import SyntheticDataset
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



class LinearInterpolator:
	@staticmethod
	def interpolate(start, end, n_steps=12):
		a, b = start.unsqueeze(1), end.unsqueeze(1)
		progress = torch.linspace(0., 1., steps=n_steps, device=a.device)\
			.view(1, n_steps, *[1]*len(a.shape[2:]))
		steps = a + (b-a)*progress
		return steps



class PathDiscriminator:
	def __init__(self, discriminator, **kwargs):
		super().__init__(**kwargs)
		self.discriminator = discriminator


	def judge(self, paths):
		B, K, *_ = paths.shape
		samples = util.combine_dims(paths, 0, 2)
		scores = self.discriminator(samples)
		return util.split_dim(scores, B, K)



class AbstractInterpolationTask(BatchedTask):
	@classmethod
	def run_step(cls, batch, info, **kwargs):
		info.clear()
		info.set_batch(batch)
		cls._encode(info)
		cls._interpolate(info)
		cls._eval_interpolations(info)
		return info


	@staticmethod
	def _encode(info):
		raise NotImplementedError


	@staticmethod
	def _interpolate(info):
		raise NotImplementedError


	@staticmethod
	def _eval_interpolations(info):
		raise NotImplementedError



class InterpolationTask(AbstractInterpolationTask):
	def __init__(self, encoder=None, interpolator=None, decoder=None,
	             path_criterion=None, discriminator=None, num_steps=12,
	             score_key=None, **kwargs):
		if score_key is None:
			score_key = 'mean'
		super().__init__(score_key=score_key, **kwargs)
		
		self.encoder = encoder
		self.interpolator = interpolator
		self.decoder = decoder
		self.path_criterion = path_criterion
		self.discriminator = discriminator
		
		self.num_steps = num_steps


	@staticmethod
	def score_names():
		return ['mean', 'std', 'min', 'max', *super().score_names()]


	@staticmethod
	def create_results_container(dataset=None, **kwargs):
		return AccumulationContainer(dataset=dataset, **kwargs)


	@staticmethod
	def _default_interpolator():
		return LinearInterpolator


	def _wrap_discriminator(self, discriminator, **kwargs):
		return PathDiscriminator(discriminator)


	def _compute(self, **kwargs):
		return super()._compute(encoder=self.encoder, interpolator=self.interpolator, decoder=self.decoder,
		                        path_criterion=self.path_criterion, discriminator=self.discriminator,
		                        num_steps=self.num_steps, **kwargs)


	@classmethod
	def prepare(cls, encoder=None, interpolator=None, decoder=None,
	            path_criterion=None, discriminator=None, num_steps=None,
	            **kwargs):
		if encoder is None:
			prt.warning('No encoder provided')
		if interpolator is None:
			prt.warning('No interpolator provided (using default)')
			interpolator = cls._default_interpolator()
		if decoder is None:
			prt.info('No decoder provided')
		if path_criterion is None:
			if discriminator is None:
				prt.warning('No path_criterion provided')
			else:
				path_criterion = cls._wrap_discriminator(discriminator)
				prt.warning('No path_criterion provided (using discriminator)')
		info = super().prepare(**kwargs)
		info.encoder = encoder
		info.interpolator = interpolator
		info.decoder = decoder
		info.path_criterion = path_criterion
		info.num_steps = num_steps
		return info


	@staticmethod
	def _encode(info):
		if 'original' not in info:
			info['original'] = batch
		code = info['original'] if info.encoder is None else info.encoder.encode(info['original'])
		info['code'] = code
		return info


	@staticmethod
	def _interpolate(info):
		if 'start' not in info or 'end' not in info:
			info['start'], info['end'] = info['code'].split(2)
		start, end = info['start'], info['end']
		info['steps'] = info.interpolator(start, end, self._num_steps)
		info['paths'] = info['steps'] if info.decoder is None \
			else util.split_dim(info.decoder.decode(util.combine_dims(info['steps'], 0, 2)), -1, self._num_steps)
		return info


	@staticmethod
	def _eval_paths(info):
		criteria = info.path_criterion(info['paths'])
		info.accumulate(criteria)
		return info
	

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







