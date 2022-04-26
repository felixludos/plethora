

from omnibelt import get_printer
prt = get_printer(__file__)

try:
	from ...community import bits_back as bb
	from . import bits_back
except (ImportError, ModuleNotFoundError):
	# raise
	prt.error('Missing bits-back dependencies for lossless compression task '
	          '(code can be found at https://github.com/bits-back/bits-back, '
	          'and then copied using the "community-source" script).')
else:
	del bb, bits_back
	from .bits_back import BitsBackCompressionTask, BitsBackCompressor

from .lossless import LosslessCompressionTask
from .lossy import LossyCompressionTask, RoundingCompressionTask
from .compressors import QuantizedCompressor, LZMACompression, SigfigCompressor, SigfigQuantizer, \
	get_lossless_compressor, get_lossy_compressor, LossyCompressor, LosslessCompressor

del prt, get_printer



# def _bits_back_download():
#
# 	files =
#
# 	pass


