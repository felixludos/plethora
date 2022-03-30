__version__ = "0.1"

import sys, os, shutil
from pathlib import Path

import omnifig as fig

from . import tasks
from . import community
from .framework import util
from .datasets import toy, mnist



_PLETHORA_COMMUNITY_PATH = Path(__file__).parents[0] / 'community'

@fig.AutoScript('community-source', description='Copy source directory to the community package')
def _download_community_directory(name, path, src_desc=None, silent=False):
	path = Path(path)
	if not path.is_dir():
		raise FileNotFoundError(str(path))

	community_path = _PLETHORA_COMMUNITY_PATH
	if not community_path.is_dir():
		community_path.mkdir(exist_ok=True)
		(community_path/'__init__.py').touch()

	dest = _PLETHORA_COMMUNITY_PATH / name
	dest.mkdir(exist_ok=True)

	if src_desc is None:
		src_desc = str(path)

	init_path = dest / '__init__.py'
	with init_path.open('a+') as f:
		f.write(f'# This directory was copied from {src_desc}\n')

	bad = {str(path / '.git'): ['objects']}
	ignore_bad_dirs = lambda d,fs: bad.get(d, [])

	shutil.copytree(str(path), str(dest), dirs_exist_ok=True, ignore=ignore_bad_dirs)
	if not silent:
		print(f'{str(path)} has been copied to the community package "{name}".')
	return dest

import torch
from tqdm import tqdm

from .datasets import MNIST
from .framework.extractors import Timm_Extractor

@fig.Script('test')
def _test_script(A):
	dataset = MNIST()
	
	model = Timm_Extractor('mobilenetv3_large_100', din=dataset.get_space('observation'))
	
	dataset.load();
	
	batch = dataset.get_batch()
	X, Y = batch['observation'], batch['target']
	
	with torch.no_grad():
		Z = model(X)
	print(X.shape, Y.shape, Z.shape)
	
	print(model)
	
	return
	
	dataset = mnist.CIFAR10().load()
	print(len(dataset))
	dataset = mnist.CIFAR10(mode='test').load()
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
		dataset = mnist.EMNIST(split=name).load()
		lns[name] = {'train': len(dataset)}
		dataset = mnist.EMNIST(split=name, mode='test').load()
		lns[name]['test'] = len(dataset)
	print(lns)
	
	
	print(len(dataset))
	
	dataset.load()
	
	print(len(dataset))
	
	print(dataset)
	
	return
	
	dataset = toy.SwissRollDataset(10, noise=.1, seed=11).load()
	
	print(list(dataset.get_iterator(batch_size=4, num_samples=5, force_batch_size=False,
                          hard_limit=False, )))
	

	return

	dataset = toy.SwissRollDataset(100).load()

	print(len(dataset))

	print(dataset.get_observation().shape)
	print(dataset.get_observation_space().sample(10).shape)

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

