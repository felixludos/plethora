import torch
from torch import nn
from torch.nn import functional as F



class Metric(nn.Module):
	def distance(self, x1, x2):
		return self(x1, x2)



class Lp_Distance(Metric, nn.PairwiseDistance):
	pass



class UnknownMetricError(Exception):
	pass



def get_metric(ident, p=2, **kwargs):
	if not isinstance(ident, str):
		return ident

	if ident.lower() == 'l1':
		return Lp_Distance(p=1, **kwargs)
	elif ident.lower() == 'l2':
		return Lp_Distance(p=2, **kwargs)
	elif ident.lower() == 'lp':
		return Lp_Distance(p=p, **kwargs)
	else:
		raise UnknownMetricError(ident)











