import copy

from omnibelt import agnosticmethod, unspecified_argument
import torch
from torch.utils.data.dataloader import default_collate
import numpy as np
from sklearn import base, metrics, cluster
# from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier

from ..util import spaces
from ..base import Function
from ..models import Model


class AbstractScikitEstimator(Model, Function):
	class ResultsContainer(Model.ResultsContainer):
		def __init__(self, estimator=None, **kwargs):
			super().__init__(**kwargs)
			self.estimator = estimator


		class UnknownResultError(KeyError):
			pass


		def get_result(self, key, **kwargs):
			# return self._find_missing(key)
			if key not in self:
				if self.estimator is None or key not in self.estimator.prediction_methods():
					raise self.UnknownResultError(key)
				self[key] = self._infer(key, **kwargs)
			return self[key]


		def _infer(self, key, observation=None, **kwargs):
			if observation is None:
				observation = self['observation']
			return self.estimator.prediction_methods()[key](observation)


		def _find_missing(self, key, **kwargs):
			if self.estimator is not None and key in self.estimator.prediction_methods():
				return self._infer(key, **kwargs)
			return super()._find_missing(key, **kwargs)


	@agnosticmethod
	def create_results_container(self, info=None, **kwargs):
		return super().create_results_container(estimator=None if type(self) == type else self, info=info, **kwargs)



	def prediction_methods(self):
		return {}


	@staticmethod
	def _format_scikit_arg(data):
		if data is not None and isinstance(data, torch.Tensor):
			data = data.detach().cpu().numpy()
		return data


	@staticmethod
	def _format_scikit_output(out):
		if out is not None and isinstance(out, np.ndarray):
			out = torch.from_numpy(out)
		return out


	@agnosticmethod
	def _get_scikit_fn(self, key):
		raise NotImplementedError


	@agnosticmethod
	def _call_scikit_fn(self, fn, *args):
		if isinstance(fn, str):
			fn = self._get_scikit_fn(fn)
		return self._format_scikit_output(fn(*[self._format_scikit_arg(arg) for arg in args]))



class AbstractSupervised(AbstractScikitEstimator): # just a flag to unify wrappers and nonwrappers
	def prediction_methods(self):
		return {'pred': self.predict}


	def predict(self, observation, **kwargs):
		return self._call_scikit_fn('predict', observation)


	def _fit(self, info, observation=None, target=None):
		if observation is None:
			observation = info['observation']
		if target is None:
			target = info['target']
		self._call_scikit_fn('fit', observation, target)
		return info



class ScikitEstimator(AbstractScikitEstimator):
	@agnosticmethod
	def _get_scikit_fn(self, key):
		return getattr(super(Model, self), key)



class ScikitEstimatorWrapper(AbstractScikitEstimator):
	def __init__(self, estimator, **kwargs):
		super().__init__(**kwargs)
		self.base_estimator = estimator


	@agnosticmethod
	def _get_scikit_fn(self, key):
		return getattr(self.base_estimator, key)



class Regressor(AbstractSupervised):
	score_key = 'r2'

	def __init__(self, *args, standardize_target=True, success_threshold=0.1, **kwargs):
		super().__init__(*args, **kwargs)
		self.standardize_target = standardize_target
		self.success_threshold = success_threshold


	def predict(self, observation, **kwargs):
		pred = super().predict(observation, **kwargs)
		if self.standardize_target:
			pred = self.dout.unstandardize(pred)
		return pred


	def _fit(self, info, observation=None, target=None):
		if target is None:
			target = info['target']
		if self.standardize_target:
			dout = info.source.space_of('target')
			target = dout.standarize(target)
		return super()._fit(info, observation=observation, target=target)


	def _evaluate(self, info):
		dout = info.source.space_of('target')

		target, pred = info['target'], info['pred']
		if self.standardize_target:
			target, pred = dout.standardize(target), dout.standardize(pred)

		# pred = torch.from_numpy(pred).float()
		# labels = torch.from_numpy(labels).float().view(*pred.shape)

		diffs = dout.difference(pred, target)
		info.diffs = diffs
		errs = diffs.abs()

		mse = errs.pow(2).mean().item()
		mae = errs.mean().item()
		mxe = errs.max().item()
		medae = errs.median().item()

		info.update({
			'error': errs,

			'mse': mse,
			'mxe': mxe,
			'mae': mae,
			'medae': medae,
		})

		# relative to prior
		if isinstance(dout, spaces.BoundDim):
			mx_error = dout.range
			if isinstance(dout, spaces.PeriodicDim):
				mx_error /= 2
			avg_error = mx_error / 2 # assuming a uniform distribution
		else:
			mx_error = target.max(0)[0] - target.min(0)[0]
			avg_error = target.std(0) # assuming a normal distribution
		mx_error, avg_error = mx_error.view(1, -1), avg_error.view(1, -1)

		info['r2'] = 1 - errs.mean(0, keepdim=True).div(avg_error).mean().item()
		if self.success_threshold is not None:
			cutoff = avg_error * self.success_threshold
			info['success'] = cutoff.sub(errs).ge(0).prod(-1).sum() / errs.shape[0]
		return info



class Classifier(AbstractSupervised):
	score_key = 'f1'


	def predict_probs(self, observation, **kwargs):
		return self._call_scikit_fn('predict_probs', observation)


	def prediction_methods(self):
		methods = super().prediction_methods()
		methods['probs'] = self.predict_probs
		return methods


	def _evaluate(self, info, **kwargs):
		dout = info.source.space_of('target')

		target, pred = info['target'], info['pred']
		target_, pred_ = self._format_scikit_arg(target.squeeze()), self._format_scikit_arg(pred.squeeze())

		report = metrics.classification_report(target_, pred_, target_names=dout.names, output_dict=True)
		confusion = metrics.confusion_matrix(target_, pred_)

		precision, recall, fscore, support = metrics.precision_recall_fscore_support(target_, pred_)

		probs_ = self._format_scikit_arg(info['probs'].squeeze())
		# multi_class = 'ovr' if
		auc = metrics.roc_auc_score(target_, probs_, multi_class='ovr')

		roc = None
		if dout.n == 2:
			roc = metrics.roc_curve(target_, probs_)

		info.update({
			'roc-auc': auc.mean(),
			'f1': fscore.mean(),
			'precision': precision.mean(),
			'recall': recall.mean(),

			'worst-roc-auc': auc.min(),
			'worst-f1': fscore.min(),
			'worst-precision': precision.min(),
			'worst-recall': recall.min(),

			'full-roc-auc': auc,
			'full-f1': fscore,
			'full-recall': recall,
			'full-precision': precision,
			'full-support': support,
			'report': report,
			'confusion': confusion,
		})
		if roc is not None:
			info['roc-curve'] = roc

		info['accuracy'] = report['accuracy']
		return info



class ParallelEstimator(AbstractScikitEstimator):
	def __init__(self, estimators, **kwargs):
		super().__init__(**kwargs)
		self.estimators = estimators


	def __getitem__(self, item):
		return self.estimators[item]


	def include_estimators(self, *estimators):
		self.estimators.extend(estimators)


	def _process_inputs(self, key, *ins, **kwargs):
		return len(self.estimators)*[ins]


	def _process_outputs(self, key, outs):
		try:
			return torch.stack(outs, 1)
		except RuntimeError:
			return outs


	def _dispatch(self, key, *ins, **kwargs):
		ins = self._process_inputs(key, *ins)
		outs = [getattr(estimator, key)(*inp) for estimator, inp in zip(self.estimators, ins)]
		return self._process_outputs(key, outs)


	def _merge_results(self, infos):
		info = self.create_results_container()
		info['individuals'] = infos
		try:
			scores = [i['score'] for i in infos]
			info['score'] = np.mean(scores)
		except KeyError:
			pass
		return infos


	def fit(self, source, **kwargs):
		infos = self._dispatch('fit', source, **kwargs)
		return self._merge_results(infos)


	def evaluate(self, source, **kwargs):
		infos = self._dispatch('evaluate', source, **kwargs)
		return self._merge_results(infos)


	def predict(self, data, **kwargs):
		return self._dispatch('predict', data, **kwargs)



class JointEstimator(ParallelEstimator): # collection of single dim estimators (can be different spaces)
	def __init__(self, estimators, **kwargs):
		super().__init__(estimators, **kwargs)
		for estimator, dout in zip(self.estimators, self.dout):
			estimator.dout = dout


	_split_key = 'target'
	def _split_source(self, key, source, split_key=None):
		if split_key is None:
			split_key = self._split_key

		target = source[split_key]

		sources = []
		start = 0
		for dim in self.dout:
			view = source.create_view()
			view.register_buffer(split_key, target.narrow(1, start, len(dim)))
			sources.append((view,))
			start += len(dim)
		return sources


	def _process_inputs(self, key, *ins, **kwargs):
		if key in {'fit', 'evaluate'}:
			return self._split_source(key, *ins, **kwargs)
		return super()._process_inputs(key, *ins, **kwargs)



class MultiEstimator(ParallelEstimator): # all estimators must be of the same kind (out space)
	def __init__(self, raw_pred=False, **kwargs):
		super().__init__(**kwargs)
		self._raw_pred = raw_pred


	def toggle_raw_pred(self, val=None):
		if val is None:
			val = not self._raw_pred
		self._raw_pred = val


	def __getattribute__(self, item):
		try:
			return super().__getattribute__(item)
		except AttributeError:
			return self.estimators[0].__getattribute__(item)


	def _merge_predictions(self, outs, **kwargs):
		raise NotImplementedError


	def _process_outputs(self, key, outs, **kwargs):
		if 'predict' in key:
			return outs if self._raw_pred else self._merge_predictions(outs, **kwargs)
		return super()._process_outputs(key, outs)



class Periodized(Regressor, MultiEstimator): # create a copy of the estimator for the sin component
	def __init__(self, estimators=None, **kwargs):
		super().__init__(estimators=estimators, **kwargs)
		assert len(self) == 2, 'must have 2 estimators for cos and sin' # TODO: use dedicated exception type


	def _prep_estimator(self, estimator): # TODO: maybe remove or fix deep copy
		assert estimator is not None, 'No base estimator provided'
		estimators = [estimator, copy.deepcopy(estimator)]
		return estimators


	_split_key = 'target'
	def _split_source(self, key, source, split_key=None):
		if split_key is None:
			split_key = self._split_key
		target = source[split_key]

		cos, sin = self.dout.expand(target).permute(2,0,1)

		cos_source = source.create_view()
		sin_source = source.create_view()

		cos_source.register_buffer(split_key, cos, space=self.estimators[0].dout)
		sin_source.register_buffer(split_key, sin, space=self.estimators[1].dout)
		return [(cos_source,), (sin_source,)]


	def _process_inputs(self, key, *ins, **kwargs):
		if key in {'fit', 'evaluate'}:
			return self._split_source(key, *ins, **kwargs)
		return super()._process_inputs(key, *ins, **kwargs)


	def evaluate(self, source, online=False, **kwargs):
		raws = None if online else super().evaluate(source)
		out = super(ParallelEstimator, self).evaluate(source)
		if raws is not None:
			out['individuals'] = raws['individuals']
		return out


	def _merge_predictions(self, outs, **kwargs):
		cos, sin = outs
		theta = self.dout.compress(torch.stack([cos, sin], -1))
		return theta



