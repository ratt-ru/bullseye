import argparse
from bullseye.helper.coord_extractor import *
from bullseye.helper.stokes import *
from bullseye.helper.channel_list_extractor import *

def build_command_line_options_parser():
  parser = argparse.ArgumentParser(description='Bullseye: An implementation of targetted facet-based synthesis imaging in radio astronomy.')
  parser.add_argument('input_ms', help='Name of the measurement set(s) to read. Multiple MSs must be comma-delimited without any separating spaces, eg. "\'one.ms\',\'two.ms\'"', type=str)
  parser.add_argument('--output_prefix', help='Prefix for the output FITS images. Facets will be indexed as [prefix_1.fits ... prefix_n.fits]', type=str, default='out.bullseye')
  parser.add_argument('--facet_centres', help='List of coordinate tupples indicating facet centres (RA,DEC).'
		      'If none are specified and n_facet_l and/or n_facet_m are not set, the default pointing centre will be used', type=coords, nargs='+', default=None)
  parser.add_argument('--npix_l', help='Number of facet pixels in l', type=int, default=256)
  parser.add_argument('--npix_m', help='Number of facet pixels in m', type=int, default=256)
  parser.add_argument('--cell_l', help='Size of a pixel in l (arcsecond)', type=float, default=1)
  parser.add_argument('--cell_m', help='Size of a pixel in m (arcsecond)', type=float, default=1)
  parser.add_argument('--pol', help='Specify image polarization', choices=pol_options.keys(), default="XX")
  parser.add_argument('--conv', help='Specify gridding convolution function type', choices=['sinc'], default='sinc')
  parser.add_argument('--conv_sup', help='Specify gridding convolution function half support area (number of convolution function cells)', type=int, default=5)
  parser.add_argument('--conv_oversamp', help='Specify gridding convolution function oversampling multiplier', type=int, default=63)
  parser.add_argument('--output_format', help='Specify image output format', choices=["fits","png"], default="fits")
  parser.add_argument('--no_chunks', help='Specify number of chunks to split measurement set into (useful to handle large measurement sets / overlap io and compute)', type=int, default=10)
  parser.add_argument('--field_id', help='Specify the id of the field (pointing) to image', type=int, default=0)
  parser.add_argument('--data_column', help='Specify the measurement set data column being imaged', type=str, default='DATA')
  parser.add_argument('--do_jones_corrections',help='Enables applying corrective jones terms per facet. Requires number of'
						    ' facet centers to be the same as the number of directions in the calibration.',type=bool,default=False)
  parser.add_argument('--n_facets_l', help='Automatically add coordinates for this number of facets in l', type=int, default=0)
  parser.add_argument('--n_facets_m', help='Automatically add coordinates for this number of facets in m', type=int, default=0)
  parser.add_argument('--stitch_facets', help='Will attempt to stitch facets together using Montage', type=bool, default=False)
  parser.add_argument('--channel_select', help="Specify list of spectral windows and channels to image, each with the format 'spw index':'comma-seperated list of channels, "
					       "for example --channel_select 0:1,3~5,7 will select channels 1,3,4,5,7 from spw 0. Default all",
		      type=channel_range, nargs='+', default=None)
  parser.add_argument('--average_spw_channels', help='Averages selected channels in each spectral window', type=bool, default=False)
  parser.add_argument('--average_all', help='Averages all selected channels together into a single image', type=bool, default=False)
  parser.add_argument('--output_psf',help='Outputs the Point Spread Function (per channel)',type=bool,default=False)
  parser.add_argument('--sample_weighting',help='Specify weighting technique in use.',choices=['natural','uniform'], default='natural')
  parser.add_argument('--open_default_viewer',help='Uses \'xdg-open\' to fire up the user\'s image viewer of choice.',default=False)
  parser.add_argument('--use_back_end',help='Switch between \'CPU\' or \'GPU\' imaging library.', choices=['CPU','GPU'], default='CPU')
  parser.add_argument('--precision',help='Force bullseye to use single / double precision when gridding', choices=['single','double'], default='single')
  parser_args = vars(parser.parse_args())
  return (parser,parser_args)