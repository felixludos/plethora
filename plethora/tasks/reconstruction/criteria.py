import torch
from torch.nn import functional as F
from omnibelt import agnosticmethod

# from pytorch_msssim import ms_ssim
import piq

from ...framework import hparam, inherit_hparams, Parametrized, Criterion
from ...framework.base import Function


class PIQ_Criterion(Criterion, Parametrized):
	pass



class PSNR(PIQ_Criterion):
	convert_to_greyscale = hparam(False)


	@agnosticmethod
	def compare(self, observation1, observation2):
		return piq.psnr(observation1, observation2, convert_to_greyscale=self.convert_to_greyscale,
		            data_range=(255 if observation1.dtype == torch.uint8 else 1.), reduction='none')



class SSIM(PIQ_Criterion):
	kernel_size = hparam(11)
	kernel_sigma = hparam(1.5)
	downsample = hparam(True)
	k1 = hparam(0.01)
	k2 = hparam(0.03)


	@agnosticmethod
	def compare(self, observation1, observation2):
		return piq.ssim(observation1, observation2,
		            kernel_size=self.kernel_size, kernel_sigma=self.kernel_sigma, downsample=self.downsample,
                    data_range=(255 if observation1.dtype == torch.uint8 else 1.), reduction='none',
                    k1=self.k1, k2=self.k2)



class MS_SSIM(PIQ_Criterion):
	kernel_size = hparam(11)
	kernel_sigma = hparam(1.5)
	scale_weights = hparam(None)
	k1 = hparam(0.01)
	k2 = hparam(0.03)
	resize = hparam(True)


	def _min_size(self):
		levels = 5 if self.scale_weights is None else len(self.scale_weights)
		return (self.kernel_size - 1) * 2 ** (levels - 1) + 1


	@agnosticmethod
	def compare(self, observation1, observation2):

		min_size = self._min_size()
		if observation1.size(-1) < min_size and self.resize:
			observation1 = F.interpolate(observation1, (min_size, min_size))
			observation2 = F.interpolate(observation2, (min_size, min_size))

		return piq.multi_scale_ssim(observation1, observation2,
		                        kernel_size=self.kernel_size, kernel_sigma=self.kernel_sigma,
		                        data_range=(255 if observation1.dtype == torch.uint8 else 1.), reduction='none',
		                        scale_weights=self.scale_weights, k1=self.k1, k2=self.k2)



class IW_SSIM(PIQ_Criterion):
	kernel_size = hparam(11)
	kernel_sigma = hparam(1.5)
	k1 = hparam(0.01)
	k2 = hparam(0.03)
	parent = hparam(True)
	blk_size = hparam(3)
	sigma_nsq = hparam(0.4)
	scale_weights = hparam(None)
	resize = hparam(True)


	def _min_size(self):
		levels = 5 if self.scale_weights is None else len(self.scale_weights)
		return (self.kernel_size - 1) * 2 ** (levels - 1) + 1


	@agnosticmethod
	def compare(self, observation1, observation2):

		min_size = self._min_size()
		if observation1.size(-1) < min_size and self.resize:
			observation1 = F.interpolate(observation1, (min_size, min_size))
			observation2 = F.interpolate(observation2, (min_size, min_size))

		return piq.information_weighted_ssim(observation1, observation2,
		                        kernel_size=self.kernel_size, kernel_sigma=self.kernel_sigma,
		                        parent=self.parent, blk_size=self.blk_size, sigma_nsq=self.sigma_nsq,
		                        data_range=(255 if observation1.dtype == torch.uint8 else 1.), reduction='none',
		                        scale_weights=self.scale_weights, k1=self.k1, k2=self.k2)



class VIFp(PIQ_Criterion):
	sigma_n_sq = hparam(2.)
	resize = hparam(True)


	@agnosticmethod
	def compare(self, observation1, observation2):
		if observation1.size(-1) < 41 and self.resize:
			observation1 = F.interpolate(observation1, (64, 64))
			observation2 = F.interpolate(observation2, (64, 64))
		return piq.vif_p(observation1, observation2, sigma_n_sq = self.sigma_n_sq,
		                        data_range=(255 if observation1.dtype == torch.uint8 else 1.), reduction='none')



class FSIM(PIQ_Criterion):
	chromatic = hparam(True)
	scales = hparam(4)
	orientations = hparam(4)
	min_length = hparam(6)
	mult = hparam(2)
	sigma_f = hparam(0.55)
	delta_theta = hparam(1.2)
	k = hparam(2.)


	@agnosticmethod
	def compare(self, observation1, observation2):
		chromatic = observation1.size(1) > 1 and self.chromatic
		return piq.fsim(observation1, observation2,
		            chromatic=chromatic, scales=self.scales, orientations=self.orientations,
		            min_length=self.min_length, mult=self.mult, sigma_f=self.sigma_f, delta_theta=self.delta_theta,
		            data_range=(255 if observation1.dtype == torch.uint8 else 1.), reduction='none')



class GMSD(PIQ_Criterion):
	t = hparam(0.00261437908496732)


	@agnosticmethod
	def compare(self, observation1, observation2):
		return 1 - piq.gmsd(observation1, observation2, t=self.t,
		            data_range=(255 if observation1.dtype == torch.uint8 else 1.), reduction='none')



class MS_GMSD(PIQ_Criterion):
	scale_weights = hparam(None)
	chromatic = hparam(False)
	alpha = hparam(0.5)
	beta1 = hparam(0.01)
	beta2 = hparam(0.32)
	beta3 = hparam(15.)
	t = hparam(170)


	@agnosticmethod
	def compare(self, observation1, observation2):
		return 1 - piq.multi_scale_gmsd(observation1, observation2,
		                                beta1=self.beta1, beta2=self.beta2, beta3=self.beta3,
		            chromatic=self.chromatic, scale_weights=self.scale_weights, t=self.t, alpha=self.alpha,
		            data_range=(255 if observation1.dtype == torch.uint8 else 1.), reduction='none')



class VSI(PIQ_Criterion):
	c1 = hparam(1.27)
	c2 = hparam(386.)
	c3 = hparam(130.)
	alpha = hparam(0.4)
	beta = hparam(0.02)
	omega_0 = hparam(0.021)
	sigma_f = hparam(1.34)
	sigma_d = hparam(145.)
	sigma_c = hparam(0.001)


	@agnosticmethod
	def compare(self, observation1, observation2):
		return piq.vsi(observation1, observation2, c1=self.c1, c2=self.c2, c3=self.c3, alpha=self.alpha, beta=self.beta,
		           omega_0=self.omega_0, sigma_f=self.sigma_f, sigma_d=self.sigma_d, sigma_c=self.sigma_c,
		            data_range=(255 if observation1.dtype == torch.uint8 else 1.), reduction='none')



class HaarPSI(PIQ_Criterion):
	scales = hparam(3)
	subsample = hparam(True)
	c = hparam(30.)
	alpha = hparam(4.2)


	@agnosticmethod
	def compare(self, observation1, observation2):
		return piq.haarpsi(observation1, observation2,
		                   scales=self.scales, subsample=self.subsample, c=self.c, alpha=self.alpha,
		            data_range=(255 if observation1.dtype == torch.uint8 else 1.), reduction='none')



class MDSI(PIQ_Criterion):
	c1 = hparam(140.)
	c2 = hparam(55.)
	c3 = hparam(550.)
	combination = hparam('sum')
	alpha = hparam(0.6)
	beta = hparam(0.1)
	gamma = hparam(0.2)
	rho = hparam(1.)
	q = hparam(0.25)
	o = hparam(0.25)


	@agnosticmethod
	def compare(self, observation1, observation2):
		return piq.mdsi(observation1, observation2,
		                c1=self.c1, c2=self.c2, c3=self.c3, alpha=self.alpha, beta=self.beta,
		           combination=self.combination, gamma=self.gamma, rho=self.rho, q=self.q, o=self.o,
		            data_range=(255 if observation1.dtype == torch.uint8 else 1.), reduction='none')



class LPIPS(PIQ_Criterion):
	replace_pooling = hparam(False)
	distance = hparam('mse', space=['mse', 'mae'])
	mean = hparam([0.485, 0.456, 0.406])
	std = hparam([0.229, 0.224, 0.225])


	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

		self.base = piq.LPIPS(replace_pooling=self.replace_pooling, distance=self.distance, reduction='none',
		                      mean=self.mean, std=self.std) # returns a loss


	def compare(self, observation1, observation2):
		return 1 - self.base(observation1, observation2)



class PieAPP(PIQ_Criterion):
	stride = hparam(27)
	resize = hparam(True)
	enable_grad = False


	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

		self.base = piq.PieAPP(reduction='none', stride=self.stride, enable_grad=self.enable_grad)


	def compare(self, observation1, observation2):
		if observation1.size(-1) != 256 and self.resize:
			observation1 = F.interpolate(observation1, (256, 256))
			observation2 = F.interpolate(observation2, (256, 256))
		return 1 - self.base(observation1, observation2)



class DISTS(PIQ_Criterion):
	mean = hparam([0.485, 0.456, 0.406])
	std = hparam([0.229, 0.224, 0.225])


	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

		self.base = piq.DISTS(reduction='none', mean=self.mean, std=self.std)


	def compare(self, observation1, observation2):
		return 1 - self.base(observation1, observation2)



class StyleScore(PIQ_Criterion):
	feature_extractor = hparam('vgg16', space=['vgg16', 'vgg19'])
	layers = hparam(('relu3_3',))
	weights = hparam([1.])
	replace_pooling = hparam(False)
	distance = hparam('mse', space=['mse', 'mae'])
	mean = hparam([0.485, 0.456, 0.406])
	std = hparam([0.229, 0.224, 0.225])
	normalize_features = hparam(False)


	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

		self.base = piq.StyleLoss(reduction='none', mean=self.mean, std=self.std, layers=self.layers,
		                          weights=self.weights, distance=self.distance,
		                          normalize_features=self.normalize_features,
		                          feature_extractor=self.feature_extractor)


	def compare(self, observation1, observation2):
		return 1 - self.base(observation1, observation2)



class ContentScore(PIQ_Criterion):
	feature_extractor = hparam('vgg16', space=['vgg16', 'vgg19'])
	layers = hparam(('relu3_3',))
	weights = hparam([1.])
	replace_pooling = hparam(False)
	distance = hparam('mse', space=['mse', 'mae'])
	mean = hparam([0.485, 0.456, 0.406])
	std = hparam([0.229, 0.224, 0.225])
	normalize_features = hparam(False)


	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

		self.base = piq.ContentLoss(reduction='none', mean=self.mean, std=self.std, layers=self.layers,
		                          weights=self.weights, distance=self.distance,
		                          normalize_features=self.normalize_features,
		                          feature_extractor=self.feature_extractor)


	def compare(self, observation1, observation2):
		return 1 - self.base(observation1, observation2)




criteria_table = {
	'psnr': PSNR,
	'ssim': SSIM,
	'msssim': MS_SSIM,
	'iwssim': IW_SSIM,
	'vifp': VIFp,
	'fsim': FSIM,
	'gmsd': GMSD,
	'msgmsd': MS_GMSD,
	'vsi': VSI,
	'haarpsi': HaarPSI,
	'mdsi': MDSI,

	'lpips': LPIPS,
	'pieapp': PieAPP,
	'dists': DISTS,
	# 'style': StyleScore,
	# 'content': ContentScore,
}
def get_criterion(ident):
	if not isinstance(ident, str):
		return ident

	if ident in criteria_table:
		return criteria_table[ident]
	fixed = ident.replace('-', '').lower()
	if fixed in criteria_table:
		return criteria_table[fixed]

	raise NotImplementedError(ident)









