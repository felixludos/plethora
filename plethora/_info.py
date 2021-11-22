



name = 'plethora'
long_name = 'Plethora Representation Suite'

version = '0.1'
url = 'https://github.com/felixludos/plethora'

description = 'A suite for evaluating representation learning methods using a plethora of different tasks'

author = 'Felix Leeb'
author_email = 'felixludos.info@gmail.com'

license = 'MIT'

readme = 'README.md'

installable_packages = ['plethora']

import os
try:
	with open(os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))), 'requirements.txt'), 'r') as f:
		install_requires = f.readlines()
except:
	install_requires = ['omnibelt','omnifig','numpy','matplotlib','torch','torchvision','tensorflow','gym','wget',
	                    'opencv-python','tabulate','ipython', 'networkx','ipdb','h5py','pyyaml','tqdm','pandas',
	                    'scikit-learn','seaborn','moviepy']
del os
