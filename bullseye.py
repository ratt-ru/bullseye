#!/usr/bin/python
import sys
import argparse
import numpy as np

from helpers import data_set_loader
from helpers import fft_utils
from helpers import convolution_filter
sys.path.append("build/algorithms")
import libimaging

def coords(s):
    try:
        ra, dec = map(float, s.split(','))
        return ra, dec
    except:
        raise argparse.ArgumentTypeError("Coordinates must be ra,dec tupples")

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Bullseye: An implementation of targetted facet-based synthesis imaging in radio astronomy.')
  pol_options = {'I': 4,'Q': 5,'U': 6,'V': 7,'XX': 0,'XY': 1,'YX': 2,'YY': 3} #if the user selects XX ,XY, YX or YY we only have to compute a single grid
  parser.add_argument('input_ms', help='Name of the measurement set to read', type=str)
  parser.add_argument('output_prefix', help='Prefix for the output FITS images. Facets will be indexed as [prefix_1.fits ... prefix_n.fits]', type=str)
  parser.add_argument('--facet_centres', help='List of coordinate tupples indicating facet centres (RA,DEC). If none present default pointing centre will be used', type=coords, nargs='+', default=None)
  parser.add_argument('--npix_l', help='Number of facet pixels in l', type=int, default=256)
  parser.add_argument('--npix_m', help='Number of facet pixels in m', type=int, default=256)
  parser.add_argument('--cell_l', help='Size of a pixel in l (arcsecond)', type=int, default=1)
  parser.add_argument('--cell_m', help='Size of a pixel in l (arcsecond)', type=int, default=1)
  parser.add_argument('--pol', help='Specify image polarization', choices=pol_options.keys(), default="XX")
  parser.add_argument('--conv', help='Specify gridding convolution function type', choices=['gausian'], default="gausian")
  parser.add_argument('--conv_sup', help='Specify gridding convolution function support area (number of grid cells)', type=int, default=1)
  parser.add_argument('--conv_oversamp', help='Specify gridding convolution function oversampling multiplier', type=int, default=1)
  parser_args = vars(parser.parse_args())
  data = data_set_loader.data_set_loader(parser_args['input_ms'])
  #some sanity checks:
  if pol_options[parser_args['pol']] > 3 and not data._no_polarization_correlations == 4:
    raise argparse.ArgumentTypeError("Cannot image polarization '%s'. Option only avaliable when data contains 4 correlation terms per visibility" % parser_args[pol])
      
  if parser_args['facet_centres'] == None: #full imaging mode with image centre == pointing centre
    conv = convolution_filter.convolution_filter(parser_args['conv_sup'],parser_args['conv_sup'],
						 parser_args['conv_oversamp'],parser_args['npix_l'],
						 parser_args['npix_m'])
    print("GRIDDING POLARIZATION %s..." % parser_args['pol']),
    g = None
    if pol_options[parser_args['pol']] < 3:
      g = libimaging.grid(data._arr_data,data._arr_uvw,
			  conv._conv_FIR.astype(np.float32),parser_args['conv_sup'],parser_args['conv_oversamp'],
			  data._no_timestamps,data._no_baselines,data._no_channels,data._no_polarization_correlations,
			  pol_options[parser_args['pol']],data._chan_wavelengths,data._arr_flaged,data._arr_flagged_rows,data._arr_weights,
			  data._phase_centre[0,0],data._phase_centre[0,1],
			  None,parser_args['npix_l'],parser_args['npix_m'],parser_args['cell_l'],parser_args['cell_m'])
    else:
      raise Exception("STUB: TODO: implement I,Q,U,V polarization options")
    print " <DONE>"
    g = np.real(fft_utils.ifft2(g[0,:,:]))/conv._F_detaper
    