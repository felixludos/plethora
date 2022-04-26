import sys, os, shutil
from pathlib import Path


PLETHORA_COMMUNITY_PATH = Path(__file__).parent
# import omnifig as fig


# @fig.AutoScript('copy-community-source', description='Copy source directory to the community package')
def copy_directory(name, path, src_desc=None, silent=False):
	path = Path(path)
	if not path.is_dir():
		raise FileNotFoundError(str(path))

	community_path = PLETHORA_COMMUNITY_PATH
	if not community_path.is_dir():
		community_path.mkdir(exist_ok=True)
		(community_path/'__init__.py').touch()

	dest = PLETHORA_COMMUNITY_PATH / name
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



# @fig.AutoScript('clone-community-source', description='Clone Github repo from the url')
def clone_repo(url, path): # TODO: add overwrite option
	from git import Repo

	path = Path(path)
	if path.exists():
		raise FileExistsError(str(path))

	repo = Repo.clone_from(url, str(path))
	repo_dir = Path(repo.common_dir).parent
	return repo_dir



def replace_file_lines():
	pass



def download_and_replace(root, name, url, manifest):

	path = root / name

	if not path.exists() or any(not (path / fname).exists() for fname in manifest):
		tmp = root / 'tmp'
		# tmp.mkdir()
		print(f'Cloning {url}')
		repo_path = clone_repo(url, str(tmp))

		path.mkdir(exist_ok=True)
		for fname, replacements in manifest.items():
			src = tmp / fname
			dest = path / fname
			if not dest.exists():
				if replacements is None:
					if src.is_dir():
						shutil.copytree(str(src), str(dest))
					else:
						shutil.copy(str(src), str(dest))
				else:
					if not dest.parent.exists():
						os.makedirs(str(dest.parent))
					dest.touch(exist_ok=True)
					with src.open('r') as fs:
						with dest.open('w') as fd:
							for line in fs.readlines():
								line = line[:-1]
								fd.write(replacements.get(line, line) + '\n')

		init_path = path / '__init__.py'
		with init_path.open('a+') as f:
			f.write(f'# This directory was copied from {url}\n')

		shutil.rmtree(tmp)

	return path



# @fig.AutoScript('get-bits-back', description='Download and prepare bits back code from Github')
def download_bits_back():
	url = 'https://github.com/bits-back/bits-back.git'

	name = 'bits_back'

	root = PLETHORA_COMMUNITY_PATH

	file_manifest = {
		'README.md': None,
		'rans.py': None,
		'util.py': {'import rans': 'from . import rans'},

		'torch_vae/torch_bin_mnist_compress.py': {
			'import util': 'from .. import util',
			'import rans': 'from .. import rans',
			'from torch_vae.tvae_binary import BinaryVAE': 'from .tvae_binary import BinaryVAE',
			'from torch_vae import tvae_utils': 'from . import tvae_utils',
		},
		'torch_vae/torch_mnist_compress.py': {
			'import util': 'from .. import util',
			'import rans': 'from .. import rans',
			'from torch_vae.tvae_beta_binomial import BetaBinomialVAE':
				'from .tvae_beta_binomial import BetaBinomialVAE',
			'from torch_vae import tvae_utils': 'from . import tvae_utils',
		},
		'torch_vae/torch_vae_binary_encode_test.py': {
			'import util': 'from .. import util',
			'import rans': 'from .. import rans',
			'from torch_vae.tvae_binary import BinaryVAE': 'from .tvae_binary import BinaryVAE',
			'from torch_vae import tvae_utils': 'from . import tvae_utils',
		},
		'torch_vae/tvae_beta_binomial.py': None,
		'torch_vae/tvae_binary.py': None,
		'torch_vae/tvae_utils.py': {
			'import util': 'from .. import util',
			'from torch_vae.tvae_beta_binomial import beta_binomial_log_pdf':
				'from .tvae_beta_binomial import beta_binomial_log_pdf',
		},
	}

	return download_and_replace(root, name, url, file_manifest)









