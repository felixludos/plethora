import torch
from torch.nn import functional as F
from omnibelt import agnosticmethod

from ...framework import hparam, inherit_hparams, ModuleParametrized, models, util


class Lp(util.Lp, models.Metric, ModuleParametrized):
	p = hparam()

# class L0(util.L0, Lp): pass
# class L1(util.L1, Lp): pass
# class L2(util.L2, Lp): pass
# class Linf(util.Linf, Lp): pass



metric_table = {
	'l0': util.L0,
	'l1': util.L1,
	'l2': util.L2,
	'linf': util.Linf,
}
def get_metric(ident):
	if isinstance(ident, (int,float)):
		return Lp(p=ident)

	if not isinstance(ident, str):
		return ident

	if ident in metric_table:
		return metric_table[ident]
	fixed = ident.replace('-', '').replace('_','').lower()
	if fixed in metric_table:
		return metric_table[fixed]

	raise NotImplementedError(ident)






