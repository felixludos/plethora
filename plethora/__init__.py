__version__ = "0.1"


from . import tasks
from . import community
from .framework import util, spaces
# from .datasets import toy, mnist
from . import datasets


# Testing


# import omnifig as fig
# import torch
# from tqdm import tqdm
#
# from omnilearn import models, util as outil
# from plethora.framework import util as putil
# from .datasets import MNIST
# from .framework import Criterion, using_rng, export, load_export
# from .framework.extractors import Timm_Extractor
# from . import tasks
# from .framework import wrapped
# # from .tasks import ReconstructionTask, FID_GenerationTask
# from plethora import datasets
# from plethora.framework import distributions
#
# from plethora.community import download_bits_back
#
# import matplotlib.pyplot as plt

# @fig.Script('test')
def _test_script(A):
	# dataset = datasets.Celeba(mode='train')#.prepare()
	#
	# print(len(dataset))
	#
	# dataset.prepare()
	#
	# print(len(dataset))
	# print(dataset.din, dataset.dout)
	#
	# batch = dataset.get_batch()
	# X, Y = batch['observation'], batch['target']
	#
	# print(X.shape)
	# print(Y)
	#
	# outil.plot_imgs(X);
	# plt.show()
	#
	# print(len(dataset))
	#
	# return


	# dataset = datasets.Shapes3D(download=False, mode='train').prepare()
	# print(dataset.fingerprint())
	# with using_rng(10):
	# 	print(dataset.fingerprint())
	# with using_rng(100):
	# 	print(dataset.fingerprint())
	# print()

	# download_bits_back()
	# return

	# mu = [-2, 0.]
	# mu = torch.zeros(3, 10)
	# sigma = .5
	#
	# mu, sigma = torch.as_tensor(mu), torch.as_tensor(sigma)
	#
	# dis = distributions.NormalDistribution(mu, sigma, seed=10)
	#
	# print(dis.sample())
	# print(dis.generate())
	#
	# m, s = dis.generate(2)
	#
	# x = distributions.Categorical(logits=s, apply_constraints=True, seed=10)
	# y = x.to('cuda')
	#
	# print(x.to('cuda'))
	#
	# print(x, repr(x))
	# print(x)
	#
	# return

	# dataset = datasets.Shapes3D(download=False, mode='test')
	#
	# dataset.prepare()
	#
	# print(len(dataset))
	#
	# return

	# print(ReconstructionTask.criterion_name)
	#
	# print(ReconstructionTask.criterion)
	#
	# ReconstructionTask.criterion = 10
	#
	# task = ReconstructionTask()
	#
	# print(task)
	#
	# return

	device = 'cuda'
	# device = 'cpu'

	dataset = datasets.Shapes3D(download=False, mode='train')
	# dataset = MNIST(batch_device=device, batch_size=200)
	len(dataset), dataset.din, dataset.dout
	dataset.prepare();
	batch = dataset.get_batch()
	len(batch)
	# obs = batch.get('observation')
	# obs.shape

	# batch = dataset.get_batch()
	# obs = batch.get('observation')
	# print(obs.shape)


	extractor = Timm_Extractor('mobilenetv3_large_100', din=dataset.observation_space)

	latent_dim = 10

	enc = wrapped.Encoder(models.make_MLP(dataset.din.shape, latent_dim).to(device),
	                      din=dataset.din, din_device=device,
	                      dout=spaces.Unbound(latent_dim), dout_device='cpu')
	dec = wrapped.Decoder(models.make_MLP(latent_dim, dataset.din.shape, output_nonlin='sigmoid').to(device),
	                      din=spaces.Unbound(latent_dim), din_device=device,
	                      dout=dataset.din, dout_device='cpu')

	def generate(N, gen=None):
		z = torch.randn(N, latent_dim, generator=gen).to(device)
		with torch.no_grad():
			return dec(z)
	dec.generate = generate


	# criterion = MSE()

	task = tasks.InferenceTask(dataset=dataset, pbar=tqdm, sample_limit=1000,
	                           # num_workers=4,
	                          encoder=enc)

	# task = tasks.PR_GenerationTask(dataset=dataset, pbar=tqdm,
	#                           sample_limit=10000, batch_size=100,
	#                           generator=dec, extractor=extractor)

	# task = tasks.ReconstructionTask(dataset=dataset, pbar=tqdm, criterion_name='ms-ssim',
	#                           sample_limit=1000, batch_size=100,
	#                           encoder=enc, decoder=dec)

	# task = tasks.RoundingCompressionTask(dataset=dataset, pbar=tqdm, sample_limit=1000, batch_size=100,
	#                                      sigfigs=3, compressor_name='lzma',
	#                                      encoder=enc, decoder=dec, criterion_name='ms-ssim')

	# task = tasks.BitsBackCompressionTask(dataset=dataset, pbar=tqdm, sample_limit=4, batch_size=2,
	#                                      encoder=enc, decoder=dec, strict_verify=True)

	# task = tasks.LosslessCompressionTask(dataset=dataset, pbar=tqdm, sample_limit=1000, batch_size=100,
	#                                      compressor_name='lzma')

	# task = tasks.CorrelationMetricTask(dataset=dataset, pbar=tqdm, metric_name='l2',
	#                           sample_limit=1000, batch_size=100,
	#                           encoder=enc)

	with torch.no_grad():
		# out = task.compute(batch)
		# print(out['score'])

		out = task.compute()
		print(out['score'])

	print(out.keys())


	return

	dataset = MNIST()
	
	model = Timm_Extractor('mobilenetv3_large_100', din=dataset.space_of('observation'))
	
	dataset.prepare();
	
	batch = dataset.get_batch()
	X, Y = batch['observation'], batch['target']
	
	with torch.no_grad():
		Z = model(X)
	print(X.shape, Y.shape, Z.shape)
	
	print(model)
	
	return
	
	dataset = mnist.CIFAR10().prepare()
	print(len(dataset))
	dataset = mnist.CIFAR10(mode='test').prepare()
	print(len(dataset))
	
	print(dataset)
	
	return
	
	classes_split_dict = {
		'byclass',
		'bymerge',
		'balanced',
		'letters',
		'digits',
		'mnist',
	}
	
	lns = {}
	for name in tqdm(classes_split_dict):
		dataset = mnist.EMNIST(split=name).prepare()
		lns[name] = {'train': len(dataset)}
		dataset = mnist.EMNIST(split=name, mode='test').prepare()
		lns[name]['test'] = len(dataset)
	print(lns)
	
	
	print(len(dataset))
	
	dataset.prepare()
	
	print(len(dataset))
	
	print(dataset)
	
	return
	
	dataset = toy.SwissRollDataset(10, noise=.1, seed=11).prepare()
	
	print(list(dataset.get_iterator(batch_size=4, sample_limit=5, force_batch_size=False,
                          hard_limit=False, )))
	

	return

	dataset = toy.SwissRollDataset(100).prepare()

	print(len(dataset))

	print(dataset.get_observation().shape)
	print(dataset.observation_space.sample(10).shape)

	print(dataset.get_label().shape)

	batches = [batch for batch in dataset]
	for batch in dataset:
		batches.append(batch)
	print([[x.shape for x in batch] for batch in batches])

	out = dataset.subset(0.5)
	print(out.get_observation().shape)

	out = dataset.split([None, 0.2])
	print(out)
	print(out.get_observation().shape)

	pass

