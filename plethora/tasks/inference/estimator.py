from omnibelt import agnosticmethod
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier

from ...framework.util import spaces
from ...framework.models import Model



class AbstractScikitModel(Model):
	
	class Builder(Model.Builder):
		def create_regressor(self, din, dout):
			raise NotImplementedError
		
	
		def create_classifier(self, din, dout):
			raise NotImplementedError
		
		
		class InvalidDimError(Exception):
			def __init__(self, din, dout):
				super().__init__(f'din={din}, dout={dout}')
				self.din, self.dout = din, dout
		
		
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





