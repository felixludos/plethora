from omnibelt import agnosticmethod
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier

from ...framework.util import spaces
from ...framework.models import Model, ModelBuilder



class AbstractSimpleBuilder(ModelBuilder):
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




class ScikitModel(Model):
	def __init__(self, estimator, standardize_target=True, **kwargs):
		super().__init__(**kwargs)
		self.estimator = estimator
		self.standardize_target = standardize_target


	# def _process_scikit_observation(self, dataset, observation=None):
	# 	if observation is None:
	# 		observation = dataset.get('observation')
	# 	return observation
	#
	#
	# def _process_scikit_target(self, dataset, target=None):
	# 	if target is None:
	# 		target = dataset.get('target')
	# 	return target


	def fit(self, dataset):
		# observation, target = self._process_scikit_observation(dataset), self._process_scikit_target(dataset)
		# observation, target = dataset.get('observation'), dataset.get('target')
		info = self.create_results_container(source=dataset)
		observation, target = info['observation'], info['target']

		space = dataset.space_of('target')
		if self.standardize_target and space is not None:
			target = space.standarize(target)

		observation, target = observation.numpy(), target.squeeze().numpy()
		self.estimator.fit(observation, target)


	def evaluate(self, dataset):
		info = self.create_results_container(source=dataset)
		observation, target = info['observation'], info['target']



		pass

	pass


class AbstractScikitModel(Model):
	
	class Builder(Model.Builder):
		def create_regressor(self, din, dout):
			raise NotImplementedError
		
	
		def create_classifier(self, din, dout):
			raise NotImplementedError
		

		def build(self, dataset, din=None, dout=None):
			if din is None:
				din = dataset.din
			if dout is None:
				dout = dataset.dout
			
			if isinstance(dout, spaces.JointSpace):
				return [self.build(dataset, din=din, dout=dim) for dim in dout]
				
			elif isinstance(dout, spaces.CategoricalDim):
				return self.create_classifier(din, dout)
			
			elif isinstance(dout, spaces.PeriodicDim):
				return self.create_regressor(din, dout)
			
			elif isinstance(dout, spaces.ContinuousDim):
				return self.create_regressor(din, dout)
			
			raise self.InvalidDimError(din=din, dout=dout)



class GradientBoostingModel(AbstractScikitModel):
	@agnosticmethod
	def create_regressor(self, din, dout):
		return GradientBoostingRegressor
	
	
	@agnosticmethod
	def create_classifier(self, din, dout):
		return GradientBoostingClassifier





