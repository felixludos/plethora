

from omnibelt import get_printer
prt = get_printer(__file__)

try:
	from ...community import bits_back as bb
except ImportError:
	prt.error('Missing bits-back dependencies for lossless compression task '
	          '(code can be found at https://github.com/bits-back/bits-back, '
	          'and then copied using the "community-source" script).')
else:
	del bb
	from .lossless import LosslessCompressionTask, BitsBackCompressionTask

del prt, get_printer

