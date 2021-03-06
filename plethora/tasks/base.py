import math as mathlib
import torch

from omnibelt import get_printer, agnosticmethod, unspecified_argument

# from ..framework.features import Seeded
from ..framework.base import Container
from ..datasets.base import Dataset
from ..framework.models import Computable
from ..framework import hparam, inherit_hparams


prt = get_printer(__file__)


class Task(Computable): # TODO: specify random seed for reproducibility
	dataset = None
	slim = False
	online = False
	auto_update_attrs = True

	score_key = 'score'

	def _extract_hparams(self, kwargs):
		remaining = super()._extract_hparams(kwargs)
		if self.auto_update_attrs:
			for key in list(remaining.keys()):
				setattr(self, key, remaining[key])
				del remaining[key]
		return remaining


	def copy(self):
		new = self.__class__()
		for key, val in self.__dict__.items():
			setattr(new, key, val)
		for key in self.iterate_hparams():
			setattr(new, key, getattr(self, key))
		return new

	
	@agnosticmethod
	def score_names(self):
		scores = super().score_names()
		if self.score_key is None:
			scores.add(self.score_key)
		return scores


	# @agnosticmethod
	# def heavy_results(self):
	# 	return set()


	@agnosticmethod
	def reset(self):
		for key in self.iterate_hparams():
			delattr(self, key)

		
	@staticmethod
	def run(info, **kwargs):
		raise NotImplementedError


	@agnosticmethod
	def filter_results(self, info):
		if self.slim:
			heavy = self.heavy_results()
			for key in heavy:
				if key in info:
					del info[key]
		if 'score' in info:
			score = info['score']
			if isinstance(score, torch.Tensor):
				score = score.item()
			info['score'] = score
		return info


	@agnosticmethod
	def compute(self, source=None, info=None, **kwargs):
		'''
		:param slim: use little space (in results)
		:param online: use little time (in computing results)
		:param seed: random seed
		:param gen: torch.Generator instance (defaults to self.gen)
		:param info: results container (created using self._create_results_container)
		:return: results container
		'''

		if isinstance(self, type): # implicit task instance
			return self(**kwargs).compute(source=source, info=info)
		# self.prepare(**kwargs)
		info = super().compute(source=source, info=info, **kwargs)
		return self.filter_results(info)



class BatchedTask(Task):
	sample_limit = None
	batch_size = None
	strict_batch_size = False
	strict_limit = True
	pbar = None


	@agnosticmethod
	def loop(self, info):
		if info.source is not None: # batch already exists (no need to generate one)
			yield info
			return

		itr = self.dataset.get_iterator(sample_limit=self.sample_limit, batch_size=self.batch_size, pbar=self.pbar,
		                                strict_batch_size=self.strict_batch_size, strict_limit=self.strict_limit,
		                                gen=info.gen)
		for batch in itr:
			info.new_source(batch)
			yield info


	@agnosticmethod
	def _compute(self, info):
		for batch in self.loop(info):
			self._compute_step(batch)
		return self.aggregate(info)


	@staticmethod
	def _compute_step(info):
		raise NotImplementedError

	
	@staticmethod
	def aggregate(info):
		return info
	


class Cumulative(BatchedTask.ResultsContainer):
	_auto_cumulative_keys = set()

	def __init__(self, cumulative_device='cpu', **kwargs):
		super().__init__(**kwargs)
		self._auto_cumulative_keys = self._auto_cumulative_keys.copy()
		# self._auto_cumulative_keys = set()
		self._cumulatives = {}
		self._cumulative_device = cumulative_device


	@agnosticmethod
	def register_cumulative(self, *keys):
		self._auto_cumulative_keys.update(keys)


	def _package_value(self, val):
		val = torch.as_tensor(val)
		if self._cumulative_device is not None and isinstance(val, torch.Tensor):
			val = val.to(self._cumulative_device)
		return val


	def accumulate(self, key, val):
		if key not in self._cumulatives:
			self._cumulatives[key] = []
		self._cumulatives[key].append(self._package_value(val))


	def auto_accumulate(self):
		for key in self._auto_cumulative_keys:
			if key in self:
				self.accumulate(key, self[key])


	def new_source(self, source): # auto accumulate
		self.auto_accumulate()
		super().new_source(source)


	def clear_cumulatives(self):
		self._cumulatives.clear()


	class MissingCumulative(KeyError):
		pass


	def aggregate(self, key, stack=False):
		if key not in self._cumulatives and (key not in self._auto_cumulative_keys or key not in self):
			raise self.MissingCumulative(key)
		elms = self._cumulatives.get(key, []).copy()
		if key in self._auto_cumulative_keys and key in self: # automatically add last batch (fence-post problem)
			elms.append(self._package_value(self[key]))
		return torch.stack(elms) if stack or len(elms[0].shape) == 0 else torch.cat(elms)



class SimpleEvaluationTask(BatchedTask):
	score_key = 'mean'
	scores_key = 'scores'


	class ResultsContainer(Cumulative, BatchedTask.ResultsContainer): # TODO: auto-accumulate scores_key
		def __init__(self, scores_key=None, **kwargs):
			super().__init__(**kwargs)
			if scores_key is not None:
				self.register_cumulative(scores_key)


	@agnosticmethod
	def create_results_container(self, info=None, scores_key=None, **kwargs):
		if scores_key is None:
			scores_key = self.scores_key
		return super().create_results_container(info=info, scores_key=scores_key, **kwargs)


	@agnosticmethod
	def score_names(self):
		return {'mean', 'std', 'min', 'max', *super().score_names()}


	@agnosticmethod
	def heavy_results(self):
		return {f'full_{self.scores_key}', *super().heavy_results()}
	

	@agnosticmethod
	def aggregate(self, info):
		info = super().aggregate(info)

		scores = info.aggregate(self.scores_key)

		info.update({
			f'full_{self.scores_key}': scores,
			'mean': scores.mean().item(),
			'max': scores.max().item(),
			'min': scores.min().item(),
			'std': scores.std().item(),
		})
		return info



class GeneralizationTask(Task):
	trial_key = 'trial'
	baseline_key = 'baseline'

	trial_score_key = 'trial_score'
	baseline_score_key = 'baseline_score'

	_bound_relative = True
	_adopt_trial_results = False

	task = hparam(module=Task)


	@hparam(module=Task)
	def baseline_task(self):
		return self.task


	@hparam(cache=True, module=Task)
	def trial_task(self):
		return self.modify_task(self.baseline_task)


	@agnosticmethod
	def modify_task(self, task):
		new = task.copy()
		new.dataset = self.modify_source(task.dataset)
		return new


	@agnosticmethod
	def unmodify_task(self, task):
		new = task.copy()
		new.dataset = self.unmodify_source(task.dataset)
		return new


	@agnosticmethod
	def modify_source(self, source):
		return source


	@agnosticmethod
	def unmodify_source(self, source):
		raise NotImplementedError


	@agnosticmethod
	def _compute(self, info):
		info = self._baseline_step(info)
		info = self._trial_step(info)
		info = self._compare_step(info)
		return info


	@agnosticmethod
	def _baseline_step(self, info):
		info[self.baseline_key] = self.baseline_task.compute(info.source)
		return info


	@agnosticmethod
	def _trial_step(self, info):
		info[self.trial_key] = self.trial_task.compute(self.modify_source(info.source))
		if self._adopt_trial_results:
			info.update(info[self.trial_key])
		return info


	@agnosticmethod
	def _compare_step(self, info):
		info[self.trial_score_key] = info[self.trial_key]['score']
		info[self.baseline_score_key] = info[self.baseline_key]['score']
		if self._bound_relative:
			info[self.trial_score_key] = min(max(0., info[self.trial_score_key]), 1.)
			info[self.baseline_score_key] = min(max(0., info[self.baseline_score_key]), 1.)
		# info[self.score_key] = info[self.trial_score_key] / (info[self.baseline_score_key] + self._score_eps)
		info[self.score_key] = mathlib.sqrt(info[self.trial_score_key] * info[self.baseline_score_key])
		return info



