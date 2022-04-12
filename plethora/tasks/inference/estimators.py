
from omnibelt import agnosticmethod, unspecified_argument

from sklearn.ensemble import GradientBoostingRegressor as _GBRegressor, GradientBoostingClassifier as _GBClassifier
from sklearn.ensemble._gb import BaseGradientBoosting
from ...framework.scikit import AbstractScikitBuilder, AbstractScikitWrapperBuilder, \
	Regressor, Classifier, ScikitEstimator


class AbstractGradientBoostingEstimator(ScikitEstimator, BaseGradientBoosting):
	@property
	def feature_importances(self):
		return self._format_scikit_output(self.feature_importances_)



class GBTRegressor(Regressor, AbstractGradientBoostingEstimator, _GBRegressor):
	pass



class GBTClassifier(Classifier, AbstractGradientBoostingEstimator, _GBClassifier):
	pass



class AbstractGradientBoostingBuilder(AbstractScikitBuilder):
	regressor_loss = 'squared_error'
	classifier_loss = 'deviance'
	learning_rate = 0.1
	n_estimators = 100
	subsample = 1.0
	criterion = 'friedman_mse'
	min_samples_split = 2
	min_samples_leaf = 1
	min_weight_fraction_leaf = 0.0
	max_depth = 3
	min_impurity_decrease = 0.0
	init = None
	random_state = None
	max_features = None
	verbose = 0
	max_leaf_nodes = None
	warm_start = False
	validation_fraction = 0.1
	n_iter_no_change = None
	tol = 0.0001
	ccp_alpha = 0.0
	alpha = 0.9



class GradientBoostingBuilder(AbstractGradientBoostingBuilder):
	def create_classifier(self, din, dout, **kwargs):
		return GBTClassifier(din=din, dout=dout, loss=self.classifier_loss, learning_rate=self.learning_rate,
		                     n_estimators=self.n_estimators, subsample=self.subsample, criterion=self.criterion,
		                     min_samples_split=self.min_samples_split, min_samples_leaf=self.min_samples_leaf,
		                     min_weight_fraction_leaf=self.min_weight_fraction_leaf, max_depth=self.max_depth,
		                     min_impurity_decrease=self.min_impurity_decrease, init=self.init,
		                     random_state=self.random_state, max_features=self.max_features, verbose=self.verbose,
		                     max_leaf_nodes=self.max_leaf_nodes, warm_start=self.warm_start,
		                     validation_fraction=self.validation_fraction, n_iter_no_change=self.n_iter_no_change,
		                     tol=self.tol, ccp_alpha=self.ccp_alpha, **kwargs)


	def create_regressor(self, din, dout, **kwargs):
		return GBTRegressor(din=din, dout=dout, loss=self.regressor_loss, learning_rate=self.learning_rate,
		                     n_estimators=self.n_estimators, subsample=self.subsample, criterion=self.criterion,
		                     min_samples_split=self.min_samples_split, min_samples_leaf=self.min_samples_leaf,
		                     min_weight_fraction_leaf=self.min_weight_fraction_leaf, max_depth=self.max_depth,
		                     min_impurity_decrease=self.min_impurity_decrease, init=self.init,
		                     random_state=self.random_state, max_features=self.max_features, verbose=self.verbose,
		                     max_leaf_nodes=self.max_leaf_nodes, warm_start=self.warm_start,
		                     validation_fraction=self.validation_fraction, n_iter_no_change=self.n_iter_no_change,
		                     tol=self.tol, ccp_alpha=self.ccp_alpha, alpha=self.alpha, **kwargs)



class GradientBoostingWrapperBuilder(AbstractScikitWrapperBuilder, AbstractGradientBoostingBuilder):
	def create_scikit_classifier(self, din, dout, **kwargs):
		return _GBClassifier(loss=self.classifier_loss, learning_rate=self.learning_rate,
		                     n_estimators=self.n_estimators, subsample=self.subsample, criterion=self.criterion,
		                     min_samples_split=self.min_samples_split, min_samples_leaf=self.min_samples_leaf,
		                     min_weight_fraction_leaf=self.min_weight_fraction_leaf, max_depth=self.max_depth,
		                     min_impurity_decrease=self.min_impurity_decrease, init=self.init,
		                     random_state=self.random_state, max_features=self.max_features, verbose=self.verbose,
		                     max_leaf_nodes=self.max_leaf_nodes, warm_start=self.warm_start,
		                     validation_fraction=self.validation_fraction, n_iter_no_change=self.n_iter_no_change,
		                     tol=self.tol, ccp_alpha=self.ccp_alpha)


	def create_scikit_regressor(self, din, dout):
		return _GBRegressor(loss=self.regressor_loss, learning_rate=self.learning_rate,
		                     n_estimators=self.n_estimators, subsample=self.subsample, criterion=self.criterion,
		                     min_samples_split=self.min_samples_split, min_samples_leaf=self.min_samples_leaf,
		                     min_weight_fraction_leaf=self.min_weight_fraction_leaf, max_depth=self.max_depth,
		                     min_impurity_decrease=self.min_impurity_decrease, init=self.init,
		                     random_state=self.random_state, max_features=self.max_features, verbose=self.verbose,
		                     max_leaf_nodes=self.max_leaf_nodes, warm_start=self.warm_start,
		                     validation_fraction=self.validation_fraction, n_iter_no_change=self.n_iter_no_change,
		                     tol=self.tol, ccp_alpha=self.ccp_alpha, alpha=self.alpha)








