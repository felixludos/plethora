__version__ = "0.1"

import sys, os, shutil
from pathlib import Path

import omnifig as fig

from . import tasks
from . import community
from .datasets import toy



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

	shutil.copytree(str(path), str(dest))
	if not silent:
		print(f'{src_desc} has been copied to the community package "{name}".')
	return dest



@fig.Script('test')
def _test_script(A):

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

