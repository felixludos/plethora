from omnibelt import agnosticmethod, unspecified_argument
import torch
from torch.utils.data.dataloader import default_collate
import numpy as np
from sklearn import base, metrics, cluster
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier

from ...datasets.buffers import NarrowBuffer
from ...datasets.base import Batch
from ...framework.util import spaces
from ...framework.base import Function
from ...framework.models import Model, ModelBuilder



class AbstractScikitBuilder(ModelBuilder):

	def create_joint(self, din, dout, estimators):
		raise NotImplementedError


	def create_regressor(self, din, dout):
		if isinstance(dout, spaces.PeriodicDim):
			return self.create_periodic(din, dout)
		return self.create_sequential(din, dout)


	def create_periodic(self, din, dout):
		raise NotImplementedError


	def create_sequential(self, din, dout):
		raise NotImplementedError


	def create_classifier(self, din, dout):
		raise NotImplementedError


	class InvalidDimError(Exception):
		def __init__(self, din, dout):
			super().__init__(f'din={din}, dout={dout}')
			self.din, self.dout = din, dout


	def build(self, din=None, dout=None):
		if din is None:
			din = getattr(self, 'din', None)
		if dout is None:
			dout = getattr(self, 'dout', None)
		if din is None or dout is None:
			dataset = getattr(self, 'dataset', None)
			if dataset is None:
				raise self.MissingKwargsError('din', 'dout', 'dataset')
			if din is None:
				din = dataset.din
			if dout is None:
				dout = dataset.dout

		if isinstance(dout, spaces.JointSpace):
			return self.create_joint(din, dout, [self.build(din=din, dout=dim) for dim in dout])

		elif isinstance(dout, spaces.CategoricalDim):
			return self.create_classifier(din, dout)

		elif isinstance(dout, spaces.ContinuousDim):
			return self.create_regressor(din, dout)

		raise self.InvalidDimError(din=din, dout=dout)



class AbstractScikitModel(Model, Function):
	@staticmethod
	def _format_scikit_arg(data):
		if data is not None and isinstance(data, torch.Tensor):
			data = data.cpu().numpy()
		return data


	@staticmethod
	def _format_scikit_output(out):
		if out is not None and isinstance(out, np.ndarray):
			out = torch.from_numpy(out)
		return out



class ScikitModel(AbstractScikitModel):
	class ResultsContainer(AbstractScikitModel.ResultsContainer):
		def __init__(self, estimator=None, **kwargs):
			super().__init__(**kwargs)
			self.estimator = estimator


		def get_result(self, key, **kwargs):
			# return self._find_missing(key)
			if key not in self:
				if self.estimator is None or key not in self.estimator.prediction_methods():
					raise self.UnknownResultError(key)
				self[key] = self._infer_missing(key, **kwargs)
			return self[key]


		def _infer(self, key, observation=None, **kwargs):
			if observation is None:
				observation = self['observation']
			return self.estimator.prediction_methods()[key](observation)


		def _find_missing(self, key, **kwargs):
			if self.estimator is not None and key in self.estimator.prediction_methods():
				self[key] = self._infer(key, **kwargs)
				return self[key]
			return super()._find_missing(key, **kwargs)


	def predict(self, observation, **kwargs):
		return self._format_scikit_output(super().predict(self._format_scikit_arg(observation), **kwargs))
	#
	#
	# def predict_score(self, observation, **kwargs):
	# 	return self._format_scikit_output(super().predict_score(self._format_scikit_arg(observation), **kwargs))
	#
	#
	# def predict_probs(self, observation, **kwargs):
	# 	return self._format_scikit_output(super().predict_probs(self._format_scikit_arg(observation), **kwargs))


	def prediction_methods(self):
		return {'pred': self.predict}
		# return {'pred': self.predict, 'scores': self.predict_score, 'probs': self.predict_probs}


	@agnosticmethod
	def create_results_container(self, info=None, **kwargs):
		return super().create_results_container(estimator=None if type(self) == type else self,
		                                        info=info, **kwargs)



class Supervised(ScikitModel):
	def _fit(self, info, observation=None, target=None):
		if observation is None:
			observation = info['observation']
		if target is None:
			target = info['target']
		super(Model, self).fit(self._format_scikit_arg(observation), self._format_scikit_arg(target.squeeze()))
		return info



class Regressor(Supervised):
	score_key = 'r2'

	def __init__(self, standardize_target=True, success_threshold=0.1, **kwargs):
		super().__init__(**kwargs)
		self.standardize_target = standardize_target
		self.success_threshold = success_threshold


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



class Classifier(Supervised):
	score_key = 'f1'


	def predict_probs(self, observation, **kwargs):
		return self._format_scikit_output(super().predict_probs(self._format_scikit_arg(observation), **kwargs))


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




class ParallelModel(AbstractScikitModel):
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
		# return outs
		try:
			return torch.stack(outs, 1)
		except RuntimeError:
			return outs
		# if 'predict' in key:
		# 	pass
		# if 'evaluate' in key:
		# 	return outs
		# return util.pytorch_collate(outs)


	def _dispatch(self, key, *ins, **kwargs):
		ins = self._process_inputs(key, *ins)
		outs = [getattr(estimator, key)(*inp) for estimator, inp in zip(self.estimators, ins)]
		return self._process_outputs(key, outs)


	def _merge_results(self, infos):
		return infos
		raise NotImplementedError
		return default_collate(infos)


	def fit(self, source, **kwargs):
		infos = self._dispatch('fit', source, **kwargs)
		return self._merge_results(infos)


	def evaluate(self, source, **kwargs):
		infos = self._dispatch('evaluate', source, **kwargs)
		return self._merge_results(infos)


	def predict(self, data, **kwargs):
		return self._dispatch('predict', data, **kwargs)



class JointEstimator(ParallelModel, ScikitModel): # collection of single dim estimators (can be different spaces)
	def __init__(self, estimators, **kwargs):
		super().__init__(estimators, **kwargs)
		for estimator, dout in zip(self.estimators, self.dout):
			estimator.dout = dout


	SelectionBuffer = NarrowBuffer
	def _split_source(self, source, idx, dim, key='target'):
		return Batch(source, buffers={key: self.SelectionBuffer(source=source.get_buffer(key), idx=idx, space=dim)})


	def _process_inputs(self, key, source, *args, **kwargs):
		if key in {'fit', 'evaluate'}:
			return [(self._split_source(source, idx, dim),) for idx, dim in enumerate(self.dout)]
		return super()._process_inputs(key, source, *args, **kwargs)


	# def _process_outputs(self, key, outs):
	# 	try:
	# 		return torch.stack(outs, 1)
	# 	except RuntimeError:
	# 		return super()._process_outputs(key, outs)



class MultiEstimator(ParallelModel): # all estimators must be of the same kind (out space)
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


	def _merge_outs(self, outs, **kwargs):
		raise NotImplementedError


	def _process_outputs(self, key, outs, **kwargs):
		if 'predict' in key:
			return outs if self._raw_pred else self._merge_outs(outs, **kwargs)
		return super()._process_outputs(key, outs)



class Periodized(MultiEstimator, Regressor): # create a copy of the estimator for the sin component
	def _process_inputs(self, key, *ins):
		if 'fit' in key:
			infos = [ScikitEstimatorInfo(est, ins[0]) for est in self.estimators]
			infos[0]._estimator_targets, infos[1]._estimator_targets = \
				self._outspace.expand(torch.from_numpy(infos[0]._estimator_targets)).permute(2,0,1).numpy()
			return [(ins[0], info) for info in infos]
		return super()._process_inputs(key, *ins)


	def evaluate(self, dataset=None, info=None, **kwargs):
		return super(ParallelEstimator, self).evaluate(dataset=dataset, info=info, **kwargs)


	def register_out_space(self, space):
		assert isinstance(space, util.PeriodicDim)
		super().register_out_space(util.JointSpace(util.BoundDim(-1,1), util.BoundDim(-1,1)))
		self._outspace = space


	def _merge_outs(self, outs, **kwargs):
		cos, sin = outs
		theta = self._outspace.compress(torch.stack([cos, sin], -1))
		return theta










# TODO: wrappers + builder for wrappers (given estimator)


class AbstractScikitModelWrapper(AbstractScikitModel):
	def __init__(self, estimator, **kwargs):
		super().__init__(**kwargs)
		self.estimator = estimator


	def _fit(self, info):
		# observation, target = self._process_scikit_observation(dataset), self._process_scikit_target(dataset)
		# observation, target = dataset.get('observation'), dataset.get('target')
		observation, target = info['observation'], info['target']
		if self.standardize_target:
			target = self.dout.standarize(target)

		self.estimator.fit(self.format_scikit_arg(observation), self.format_scikit_arg(target.squeeze()))
		return info


	def predict(self, observation):
		pred = self.estimator.predict(self.format_scikit_arg(observation))
		if len(pred.shape) == 1:
			pred = pred.reshape(-1, 1)
		pred = self.format_scikit_output(pred)
		if self.standardize_target:
			pred = self.dout.unstandardize(pred)
		return pred


# TODO: AbstractScikitModelWrapper subclasses for different types








