#!/usr/bin/python
import sys
import argparse
import numpy as np
import pylab
import re
from pyrap.quanta import quantity

from helpers import data_set_loader
from helpers import fft_utils
from helpers import convolution_filter
from helpers import fits_export
from helpers import base_types
import ctypes
libimaging = ctypes.pydll.LoadLibrary("build/algorithms/libimaging.so")
	   
def coords(s):  
    try:
	sT = s.strip()
        ra, dec = map(float, sT[1:len(sT)-1].split(','))
        return ra, dec
    except:
        raise argparse.ArgumentTypeError("Coordinates must be ra,dec tupples")

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Bullseye: An implementation of targetted facet-based synthesis imaging in radio astronomy.')
  pol_options = {'I' : 1, 'Q' : 2, 'U' : 3, 'V' : 4, 'RR' : 5, 'RL' : 6, 'LR' : 7, 'LL' : 8, 'XX' : 9, 'XY' : 10, 'YX' : 11, 'YY' : 12} # as per Stokes.h in casacore, the rest is left unimplemented
  parser.add_argument('input_ms', help='Name of the measurement set to read', type=str)
  parser.add_argument('output_prefix', help='Prefix for the output FITS images. Facets will be indexed as [prefix_1.fits ... prefix_n.fits]', type=str)
  parser.add_argument('--facet_centres', help='List of coordinate tupples indicating facet centres (RA,DEC). If none present default pointing centre will be used', type=coords, nargs='+', default=None)
  parser.add_argument('--npix_l', help='Number of facet pixels in l', type=int, default=256)
  parser.add_argument('--npix_m', help='Number of facet pixels in m', type=int, default=256)
  parser.add_argument('--cell_l', help='Size of a pixel in l (arcsecond)', type=float, default=1)
  parser.add_argument('--cell_m', help='Size of a pixel in l (arcsecond)', type=float, default=1)
  parser.add_argument('--pol', help='Specify image polarization', choices=pol_options.keys(), default="XX")
  parser.add_argument('--conv', help='Specify gridding convolution function type', choices=['gausian','keiser bessel'], default='keiser bessel')
  parser.add_argument('--conv_sup', help='Specify gridding convolution function support area (number of grid cells)', type=int, default=1)
  parser.add_argument('--conv_oversamp', help='Specify gridding convolution function oversampling multiplier', type=int, default=1)
  parser.add_argument('--output_format', help='Specify image output format', choices=["fits","png"], default="fits")
  parser.add_argument('--mem_available_for_input_data', help='Specify available memory (bytes) for storing the input measurement set data arrays', type=int, default=512*1024*1024)
  parser.add_argument('--field_id', help='Specify the id of the field (pointing) to image', type=int, default=0)
  parser.add_argument('--data_column', help='Specify the measurement set data column being imaged', type=str, default='DATA')
  parser.add_argument('--do_jones_corrections',help='Enables applying corrective jones terms per facet. Requires number of'
						    ' facet centers to be the same as the number of directions in the calibration.',type=bool,default=False)
  parser.add_argument('--output_psf',help='Outputs the point-spread-function',type=bool,default=False)
  parser_args = vars(parser.parse_args())
  data = data_set_loader.data_set_loader(parser_args['input_ms'],read_jones_terms=parser_args['do_jones_corrections'])
  data.read_head()
  chunk_size = data.compute_number_of_rows_to_read_from_mem_requirements(parser_args['mem_available_for_input_data'])
  if chunk_size == 0:
    raise Exception("Insufficient memory allocated for loading data. Cannot even load a single row and a timestamp of jones matricies at a time")
  no_chunks = data.number_of_read_iterations_required_from_mem_requirements(parser_args['mem_available_for_input_data'])
  #some sanity checks:
  if (parser_args['pol'] in ['I','Q','U','V'] and 
      pol_options[parser_args['pol']] not in data._polarization_correlations and 
      data._polarization_correlations.tolist() not in [[pol_options['RR'],pol_options['RL'],pol_options['LR'],pol_options['LL']],	#circular correlation products
						       [pol_options['XX'],pol_options['XY'],pol_options['YX'],pol_options['YY']]]):	#linear correlation products
      raise argparse.ArgumentTypeError("Unsupported polarization option specified in Measurement Set. Stokes terms "
				       "may only be derived from ['RR','RL','LR','LL'] or ['XX','XY','YX','YY']")
  
  elif (parser_args['pol'] not in ['I','Q','U','V']) and (pol_options[parser_args['pol']] not in data._polarization_correlations):
    raise argparse.ArgumentTypeError("Cannot obtain requested gridded polarization from the provided measurement set.")
  if parser_args['field_id'] not in range(0,len(data._field_centre_names)):
    raise argparse.ArgumentTypeError("Specified field does not exist Must be in 0 ... %d for this Measurement Set" % (len(data._field_centre_names) - 1))
  if parser_args['do_jones_corrections'] and data._no_polarization_correlations != 4:
    raise argparse.ArgumentTypeError("Measurement set must contain 4 correlation terms per visibility in order to apply corrective jones matricies")
  if parser_args['do_jones_corrections'] and (not data._dde_cal_info_exists or not data._dde_cal_info_desc_exists):
    raise argparse.ArgumentTypeError("Measurement set does not contain corrective DDE terms or the description table is missing.")
  
  conv = convolution_filter.convolution_filter(parser_args['conv_sup'],parser_args['conv_sup'],
					       parser_args['conv_oversamp'],parser_args['npix_l'],
					       parser_args['npix_m'],parser_args['conv'])
  print "IMAGING ONLY FIELD %s" % data._field_centre_names[parser_args['field_id']]
  facet_centres = None
  
  num_facet_centres = 0
  if (parser_args['facet_centres'] != None):
    num_facet_centres = len(parser_args['facet_centres'])
    facet_centres = np.array(parser_args['facet_centres']).astype(base_types.uvw_type)
  if parser_args['do_jones_corrections'] and num_facet_centres != data._cal_no_dirs:
    raise argparse.ArgumentTypeError("Number of calibrated directions does not correspond to number of directions being faceted")
  
  gridded_vis = None
  gridded_sampling_function = None
  #no need to grid more than one of the correlations if the user isn't interrested in imaging one of the stokes terms (I,Q,U,V) or the stokes terms are the correlation products:
  if pol_options[parser_args['pol']] in data._polarization_correlations.tolist() and not parser_args['do_jones_corrections']:
    pol_index = pol_options[parser_args['pol']]
    pol_index = data._polarization_correlations.tolist().index(pol_options[parser_args['pol']])
    
    num_grids = 1 if (num_facet_centres == 0) else num_facet_centres
    g = np.zeros([num_grids,1,parser_args['npix_l'],parser_args['npix_m']],dtype=base_types.grid_type)
    sampling_funct = np.zeros([num_grids,1,parser_args['npix_l'],parser_args['npix_m']],dtype=base_types.psf_type)
    for chunk_index in range(0,no_chunks):
      chunk_lbound = chunk_index * chunk_size
      chunk_ubound = min((chunk_index+1) * chunk_size,data._no_rows)
      chunk_linecount = chunk_ubound - chunk_lbound
      print "READING CHUNK %d OF %d" % (chunk_index+1,no_chunks)
      data.read_data(start_row=chunk_lbound,no_rows=chunk_linecount,data_column = parser_args['data_column'])
      if (parser_args['facet_centres'] == None):
	if parser_args['output_psf']:
	  libimaging.grid_single_pol_with_sampling_func(data._arr_data.ctypes.data_as(ctypes.c_void_p),
				     data._arr_uvw.ctypes.data_as(ctypes.c_void_p),
				     ctypes.c_size_t(data._no_timestamps_read),ctypes.c_size_t(data._no_baselines),
				     ctypes.c_size_t(data._no_channels),ctypes.c_size_t(data._no_polarization_correlations),
				     data._chan_wavelengths.ctypes.data_as(ctypes.c_void_p),
				     data._arr_flaged.ctypes.data_as(ctypes.c_void_p),
				     data._arr_flagged_rows.ctypes.data_as(ctypes.c_void_p),
				     data._arr_weights.ctypes.data_as(ctypes.c_void_p),
				     ctypes.c_size_t(parser_args['npix_l']),
				     ctypes.c_size_t(parser_args['npix_m']),
				     base_types.uvw_ctypes_convert_type(parser_args['cell_l']),
				     base_types.uvw_ctypes_convert_type(parser_args['cell_m']),
				     conv._conv_FIR.ctypes.data_as(ctypes.c_void_p),
				     ctypes.c_size_t(parser_args['conv_sup']),
				     ctypes.c_size_t(parser_args['conv_oversamp']),
				     ctypes.c_size_t(pol_index),
				     g.ctypes.data_as(ctypes.c_void_p),
				     ctypes.c_size_t(chunk_linecount),
				     data._row_field_id.ctypes.data_as(ctypes.c_void_p),
				     ctypes.c_uint(parser_args['field_id']),
				     data._description_col.ctypes.data_as(ctypes.c_void_p),
				     sampling_funct.ctypes.data_as(ctypes.c_void_p))
	else:
	  libimaging.grid_single_pol(data._arr_data.ctypes.data_as(ctypes.c_void_p),
				     data._arr_uvw.ctypes.data_as(ctypes.c_void_p),
				     ctypes.c_size_t(data._no_timestamps_read),ctypes.c_size_t(data._no_baselines),
				     ctypes.c_size_t(data._no_channels),ctypes.c_size_t(data._no_polarization_correlations),
				     data._chan_wavelengths.ctypes.data_as(ctypes.c_void_p),
				     data._arr_flaged.ctypes.data_as(ctypes.c_void_p),
				     data._arr_flagged_rows.ctypes.data_as(ctypes.c_void_p),
				     data._arr_weights.ctypes.data_as(ctypes.c_void_p),
				     ctypes.c_size_t(parser_args['npix_l']),
				     ctypes.c_size_t(parser_args['npix_m']),
				     base_types.uvw_ctypes_convert_type(parser_args['cell_l']),
				     base_types.uvw_ctypes_convert_type(parser_args['cell_m']),
				     conv._conv_FIR.ctypes.data_as(ctypes.c_void_p),
				     ctypes.c_size_t(parser_args['conv_sup']),
				     ctypes.c_size_t(parser_args['conv_oversamp']),
				     ctypes.c_size_t(pol_index),
				     g.ctypes.data_as(ctypes.c_void_p),
				     ctypes.c_size_t(chunk_linecount),
				     data._row_field_id.ctypes.data_as(ctypes.c_void_p),
				     ctypes.c_uint(parser_args['field_id']),
				     data._description_col.ctypes.data_as(ctypes.c_void_p))
      else:
	if parser_args['output_psf']:
	  libimaging.facet_single_pol_with_sampling_func(data._arr_data.ctypes.data_as(ctypes.c_void_p),
				      data._arr_uvw.ctypes.data_as(ctypes.c_void_p),
				      ctypes.c_size_t(data._no_timestamps_read),ctypes.c_size_t(data._no_baselines),
				      ctypes.c_size_t(data._no_channels),ctypes.c_size_t(data._no_polarization_correlations),
				      data._chan_wavelengths.ctypes.data_as(ctypes.c_void_p),
				      data._arr_flaged.ctypes.data_as(ctypes.c_void_p),
				      data._arr_flagged_rows.ctypes.data_as(ctypes.c_void_p),
				      data._arr_weights.ctypes.data_as(ctypes.c_void_p),
				      ctypes.c_size_t(parser_args['npix_l']),
				      ctypes.c_size_t(parser_args['npix_m']),
				      base_types.uvw_ctypes_convert_type(parser_args['cell_l']),
				      base_types.uvw_ctypes_convert_type(parser_args['cell_m']),
				      base_types.uvw_ctypes_convert_type(data._field_centres[parser_args['field_id'],0,0]),
				      base_types.uvw_ctypes_convert_type(data._field_centres[parser_args['field_id'],0,1]),
				      facet_centres.ctypes.data_as(ctypes.c_void_p) if (num_facet_centres != 0) else None, 
				      ctypes.c_size_t(num_facet_centres), 
				      conv._conv_FIR.ctypes.data_as(ctypes.c_void_p),
				      ctypes.c_size_t(parser_args['conv_sup']),
				      ctypes.c_size_t(parser_args['conv_oversamp']),
				      ctypes.c_size_t(pol_index),
				      g.ctypes.data_as(ctypes.c_void_p),
				      ctypes.c_size_t(chunk_linecount),
				      data._row_field_id.ctypes.data_as(ctypes.c_void_p),
				      ctypes.c_uint(parser_args['field_id']),
				      data._description_col.ctypes.data_as(ctypes.c_void_p),
				      sampling_funct.ctypes.data_as(ctypes.c_void_p))
	else:
	  libimaging.facet_single_pol(data._arr_data.ctypes.data_as(ctypes.c_void_p),
				      data._arr_uvw.ctypes.data_as(ctypes.c_void_p),
				      ctypes.c_size_t(data._no_timestamps_read),ctypes.c_size_t(data._no_baselines),
				      ctypes.c_size_t(data._no_channels),ctypes.c_size_t(data._no_polarization_correlations),
				      data._chan_wavelengths.ctypes.data_as(ctypes.c_void_p),
				      data._arr_flaged.ctypes.data_as(ctypes.c_void_p),
				      data._arr_flagged_rows.ctypes.data_as(ctypes.c_void_p),
				      data._arr_weights.ctypes.data_as(ctypes.c_void_p),
				      ctypes.c_size_t(parser_args['npix_l']),
				      ctypes.c_size_t(parser_args['npix_m']),
				      base_types.uvw_ctypes_convert_type(parser_args['cell_l']),
				      base_types.uvw_ctypes_convert_type(parser_args['cell_m']),
				      base_types.uvw_ctypes_convert_type(data._field_centres[parser_args['field_id'],0,0]),
				      base_types.uvw_ctypes_convert_type(data._field_centres[parser_args['field_id'],0,1]),
				      facet_centres.ctypes.data_as(ctypes.c_void_p) if (num_facet_centres != 0) else None, 
				      ctypes.c_size_t(num_facet_centres), 
				      conv._conv_FIR.ctypes.data_as(ctypes.c_void_p),
				      ctypes.c_size_t(parser_args['conv_sup']),
				      ctypes.c_size_t(parser_args['conv_oversamp']),
				      ctypes.c_size_t(pol_index),
				      g.ctypes.data_as(ctypes.c_void_p),
				      ctypes.c_size_t(chunk_linecount),
				      data._row_field_id.ctypes.data_as(ctypes.c_void_p),
				      ctypes.c_uint(parser_args['field_id']),
				      data._description_col.ctypes.data_as(ctypes.c_void_p))
      
    gridded_vis = g[:,0,:,:]
  else: # the user want to derive one of the stokes terms (I,Q,U,V) from the correlation terms:
    num_polarized_grids = 1 if (num_facet_centres == 0) else num_facet_centres
    g = np.zeros([num_polarized_grids,4,parser_args['npix_l'],parser_args['npix_m']],dtype=base_types.grid_type)
    sampling_funct = np.zeros([num_polarized_grids,1,parser_args['npix_l'],parser_args['npix_m']],dtype=base_types.psf_type) #only one Sampling function over all polarizations
    for chunk_index in range(0,no_chunks):
      chunk_lbound = chunk_index * chunk_size
      chunk_ubound = min((chunk_index+1) * chunk_size,data._no_rows)
      chunk_linecount = chunk_ubound - chunk_lbound
      print "READING CHUNK %d OF %d" % (chunk_index+1,no_chunks)
      data.read_data(start_row=chunk_lbound,no_rows=chunk_linecount,data_column = parser_args['data_column'])
      if (parser_args['facet_centres'] == None):
	if parser_args['output_psf']:
	  libimaging.grid_4_cor_with_sampling_func(data._arr_data.ctypes.data_as(ctypes.c_void_p),
				data._arr_uvw.ctypes.data_as(ctypes.c_void_p),
				ctypes.c_size_t(data._no_timestamps_read),ctypes.c_size_t(data._no_baselines),
				ctypes.c_size_t(data._no_channels),ctypes.c_size_t(data._no_polarization_correlations),
				data._chan_wavelengths.ctypes.data_as(ctypes.c_void_p),
				data._arr_flaged.ctypes.data_as(ctypes.c_void_p),
				data._arr_flagged_rows.ctypes.data_as(ctypes.c_void_p),
				data._arr_weights.ctypes.data_as(ctypes.c_void_p),
				ctypes.c_size_t(parser_args['npix_l']),
				ctypes.c_size_t(parser_args['npix_m']),
				base_types.uvw_ctypes_convert_type(parser_args['cell_l']),
				base_types.uvw_ctypes_convert_type(parser_args['cell_m']),
				conv._conv_FIR.ctypes.data_as(ctypes.c_void_p),
				ctypes.c_size_t(parser_args['conv_sup']),
				ctypes.c_size_t(parser_args['conv_oversamp']),
				g.ctypes.data_as(ctypes.c_void_p),
				ctypes.c_size_t(chunk_linecount),
				data._row_field_id.ctypes.data_as(ctypes.c_void_p),
				ctypes.c_uint(parser_args['field_id']),
				data._description_col.ctypes.data_as(ctypes.c_void_p),
				sampling_funct.ctypes.data_as(ctypes.c_void_p))
	else:
	  libimaging.grid_4_cor(data._arr_data.ctypes.data_as(ctypes.c_void_p),
				data._arr_uvw.ctypes.data_as(ctypes.c_void_p),
				ctypes.c_size_t(data._no_timestamps_read),ctypes.c_size_t(data._no_baselines),
				ctypes.c_size_t(data._no_channels),ctypes.c_size_t(data._no_polarization_correlations),
				data._chan_wavelengths.ctypes.data_as(ctypes.c_void_p),
				data._arr_flaged.ctypes.data_as(ctypes.c_void_p),
				data._arr_flagged_rows.ctypes.data_as(ctypes.c_void_p),
				data._arr_weights.ctypes.data_as(ctypes.c_void_p),
				ctypes.c_size_t(parser_args['npix_l']),
				ctypes.c_size_t(parser_args['npix_m']),
				base_types.uvw_ctypes_convert_type(parser_args['cell_l']),
				base_types.uvw_ctypes_convert_type(parser_args['cell_m']),
				conv._conv_FIR.ctypes.data_as(ctypes.c_void_p),
				ctypes.c_size_t(parser_args['conv_sup']),
				ctypes.c_size_t(parser_args['conv_oversamp']),
				g.ctypes.data_as(ctypes.c_void_p),
				ctypes.c_size_t(chunk_linecount),
				data._row_field_id.ctypes.data_as(ctypes.c_void_p),
				ctypes.c_uint(parser_args['field_id']),
				data._description_col.ctypes.data_as(ctypes.c_void_p))
      elif parser_args['do_jones_corrections']:
	if parser_args['output_psf']:
	  libimaging.facet_4_cor_corrections_with_sampling_func(data._arr_data.ctypes.data_as(ctypes.c_void_p),
					     data._arr_uvw.ctypes.data_as(ctypes.c_void_p),
					     ctypes.c_size_t(data._no_timestamps_read),ctypes.c_size_t(data._no_baselines),
					     ctypes.c_size_t(data._no_channels),ctypes.c_size_t(data._no_polarization_correlations),
					     data._chan_wavelengths.ctypes.data_as(ctypes.c_void_p),
					     data._arr_flaged.ctypes.data_as(ctypes.c_void_p),
					     data._arr_flagged_rows.ctypes.data_as(ctypes.c_void_p),
					     data._arr_weights.ctypes.data_as(ctypes.c_void_p),
					     ctypes.c_size_t(parser_args['npix_l']),
					     ctypes.c_size_t(parser_args['npix_m']),
					     base_types.uvw_ctypes_convert_type(parser_args['cell_l']),
					     base_types.uvw_ctypes_convert_type(parser_args['cell_m']),
					     base_types.uvw_ctypes_convert_type(data._field_centres[parser_args['field_id'],0,0]),
					     base_types.uvw_ctypes_convert_type(data._field_centres[parser_args['field_id'],0,1]),
					     facet_centres.ctypes.data_as(ctypes.c_void_p) if (num_facet_centres != 0) else None, 
					     ctypes.c_size_t(num_facet_centres), 
					     conv._conv_FIR.ctypes.data_as(ctypes.c_void_p),
					     ctypes.c_size_t(parser_args['conv_sup']),
					     ctypes.c_size_t(parser_args['conv_oversamp']),
					     g.ctypes.data_as(ctypes.c_void_p),
					     ctypes.c_size_t(chunk_linecount),
					     data._row_field_id.ctypes.data_as(ctypes.c_void_p),
					     ctypes.c_uint(parser_args['field_id']),
					     data._description_col.ctypes.data_as(ctypes.c_void_p),
					     data._jones_terms.ctypes.data_as(ctypes.c_void_p),
					     ctypes.c_bool(True),
					     data._arr_antenna_1.ctypes.data_as(ctypes.c_void_p),
					     data._arr_antenna_2.ctypes.data_as(ctypes.c_void_p),
					     data._time_indicies.ctypes.data_as(ctypes.c_void_p),
					     ctypes.c_size_t(data._no_antennae),
					     ctypes.c_size_t(data._no_timestamps_read),
					     ctypes.c_size_t(data._no_spw),
					     sampling_funct.ctypes.data_as(ctypes.c_void_p))
	else:
	  libimaging.facet_4_cor_corrections(data._arr_data.ctypes.data_as(ctypes.c_void_p),
					     data._arr_uvw.ctypes.data_as(ctypes.c_void_p),
					     ctypes.c_size_t(data._no_timestamps_read),ctypes.c_size_t(data._no_baselines),
					     ctypes.c_size_t(data._no_channels),ctypes.c_size_t(data._no_polarization_correlations),
					     data._chan_wavelengths.ctypes.data_as(ctypes.c_void_p),
					     data._arr_flaged.ctypes.data_as(ctypes.c_void_p),
					     data._arr_flagged_rows.ctypes.data_as(ctypes.c_void_p),
					     data._arr_weights.ctypes.data_as(ctypes.c_void_p),
					     ctypes.c_size_t(parser_args['npix_l']),
					     ctypes.c_size_t(parser_args['npix_m']),
					     base_types.uvw_ctypes_convert_type(parser_args['cell_l']),
					     base_types.uvw_ctypes_convert_type(parser_args['cell_m']),
					     base_types.uvw_ctypes_convert_type(data._field_centres[parser_args['field_id'],0,0]),
					     base_types.uvw_ctypes_convert_type(data._field_centres[parser_args['field_id'],0,1]),
					     facet_centres.ctypes.data_as(ctypes.c_void_p) if (num_facet_centres != 0) else None, 
					     ctypes.c_size_t(num_facet_centres), 
					     conv._conv_FIR.ctypes.data_as(ctypes.c_void_p),
					     ctypes.c_size_t(parser_args['conv_sup']),
					     ctypes.c_size_t(parser_args['conv_oversamp']),
					     g.ctypes.data_as(ctypes.c_void_p),
					     ctypes.c_size_t(chunk_linecount),
					     data._row_field_id.ctypes.data_as(ctypes.c_void_p),
					     ctypes.c_uint(parser_args['field_id']),
					     data._description_col.ctypes.data_as(ctypes.c_void_p),
					     data._jones_terms.ctypes.data_as(ctypes.c_void_p),
					     ctypes.c_bool(True),
					     data._arr_antenna_1.ctypes.data_as(ctypes.c_void_p),
					     data._arr_antenna_2.ctypes.data_as(ctypes.c_void_p),
					     data._time_indicies.ctypes.data_as(ctypes.c_void_p),
					     ctypes.c_size_t(data._no_antennae),
					     ctypes.c_size_t(data._no_timestamps_read),
					     ctypes.c_size_t(data._no_spw))
      else:
	if parser_args['output_psf']:
	  libimaging.facet_4_cor_with_sampling_func(data._arr_data.ctypes.data_as(ctypes.c_void_p),
				 data._arr_uvw.ctypes.data_as(ctypes.c_void_p),
				 ctypes.c_size_t(data._no_timestamps_read),ctypes.c_size_t(data._no_baselines),
				 ctypes.c_size_t(data._no_channels),ctypes.c_size_t(data._no_polarization_correlations),
				 data._chan_wavelengths.ctypes.data_as(ctypes.c_void_p),
				 data._arr_flaged.ctypes.data_as(ctypes.c_void_p),
				 data._arr_flagged_rows.ctypes.data_as(ctypes.c_void_p),
				 data._arr_weights.ctypes.data_as(ctypes.c_void_p),
				 ctypes.c_size_t(parser_args['npix_l']),
				 ctypes.c_size_t(parser_args['npix_m']),
				 base_types.uvw_ctypes_convert_type(parser_args['cell_l']),
				 base_types.uvw_ctypes_convert_type(parser_args['cell_m']),
				 base_types.uvw_ctypes_convert_type(data._field_centres[parser_args['field_id'],0,0]),
				 base_types.uvw_ctypes_convert_type(data._field_centres[parser_args['field_id'],0,1]),
				 facet_centres.ctypes.data_as(ctypes.c_void_p) if (num_facet_centres != 0) else None, 
				 ctypes.c_size_t(num_facet_centres), 
				 conv._conv_FIR.ctypes.data_as(ctypes.c_void_p),
				 ctypes.c_size_t(parser_args['conv_sup']),
				 ctypes.c_size_t(parser_args['conv_oversamp']),
				 g.ctypes.data_as(ctypes.c_void_p),
				 ctypes.c_size_t(chunk_linecount),
				 data._row_field_id.ctypes.data_as(ctypes.c_void_p),
				 ctypes.c_uint(parser_args['field_id']),
				 data._description_col.ctypes.data_as(ctypes.c_void_p),
				 sampling_funct.ctypes.data_as(ctypes.c_void_p))
	else:
	  libimaging.facet_4_cor(data._arr_data.ctypes.data_as(ctypes.c_void_p),
				 data._arr_uvw.ctypes.data_as(ctypes.c_void_p),
				 ctypes.c_size_t(data._no_timestamps_read),ctypes.c_size_t(data._no_baselines),
				 ctypes.c_size_t(data._no_channels),ctypes.c_size_t(data._no_polarization_correlations),
				 data._chan_wavelengths.ctypes.data_as(ctypes.c_void_p),
				 data._arr_flaged.ctypes.data_as(ctypes.c_void_p),
				 data._arr_flagged_rows.ctypes.data_as(ctypes.c_void_p),
				 data._arr_weights.ctypes.data_as(ctypes.c_void_p),
				 ctypes.c_size_t(parser_args['npix_l']),
				 ctypes.c_size_t(parser_args['npix_m']),
				 base_types.uvw_ctypes_convert_type(parser_args['cell_l']),
				 base_types.uvw_ctypes_convert_type(parser_args['cell_m']),
				 base_types.uvw_ctypes_convert_type(data._field_centres[parser_args['field_id'],0,0]),
				 base_types.uvw_ctypes_convert_type(data._field_centres[parser_args['field_id'],0,1]),
				 facet_centres.ctypes.data_as(ctypes.c_void_p) if (num_facet_centres != 0) else None, 
				 ctypes.c_size_t(num_facet_centres), 
				 conv._conv_FIR.ctypes.data_as(ctypes.c_void_p),
				 ctypes.c_size_t(parser_args['conv_sup']),
				 ctypes.c_size_t(parser_args['conv_oversamp']),
				 g.ctypes.data_as(ctypes.c_void_p),
				 ctypes.c_size_t(chunk_linecount),
				 data._row_field_id.ctypes.data_as(ctypes.c_void_p),
				 ctypes.c_uint(parser_args['field_id']),
				 data._description_col.ctypes.data_as(ctypes.c_void_p))
    '''
    See Smirnov I (2011) for description on conversion between correlation terms and stokes params for linearly polarized feeds
    See Synthesis Imaging II (1999) Pg. 9 for a description on conversion between correlation terms and stokes params for circularly polarized feeds
    '''
    if data._polarization_correlations.tolist() == [pol_options['RR'],pol_options['RL'],pol_options['LR'],pol_options['LL']]:		#circular correlation products
      if parser_args['pol'] == "I":
	gridded_vis = ((g[:,0,:,:] + g[:,3,:,:])/2).reshape(num_polarized_grids,parser_args['npix_l'],parser_args['npix_m'])
      elif parser_args['pol'] == "V":
	gridded_vis = ((g[:,0,:,:] - g[:,3,:,:])/2).reshape(num_polarized_grids,parser_args['npix_l'],parser_args['npix_m'])
      elif parser_args['pol'] == "Q":
	gridded_vis = ((g[:,1,:,:] + g[:,2,:,:])/2).reshape(num_polarized_grids,parser_args['npix_l'],parser_args['npix_m'])
      elif parser_args['pol'] == "U":
	gridded_vis = ((g[:,1,:,:] - g[:,2,:,:])/2.0).reshape(num_polarized_grids,parser_args['npix_l'],parser_args['npix_m'])
    elif data._polarization_correlations.tolist() == [pol_options['XX'],pol_options['XY'],pol_options['YX'],pol_options['YY']]:		#linear correlation products
      if parser_args['pol'] == "I":
	gridded_vis = ((g[:,0,:,:] + g[:,3,:,:])).reshape(num_polarized_grids,parser_args['npix_l'],parser_args['npix_m'])
      elif parser_args['pol'] == "V":
	gridded_vis = ((g[:,1,:,:] - g[:,2,:,:])/1.0).reshape(num_polarized_grids,parser_args['npix_l'],parser_args['npix_m'])
      elif parser_args['pol'] == "Q":
	gridded_vis = ((g[:,0,:,:] - g[:,3,:,:])).reshape(num_polarized_grids,parser_args['npix_l'],parser_args['npix_m'])
      elif parser_args['pol'] == "U":
	gridded_vis = ((g[:,1,:,:] + g[:,2,:,:])).reshape(num_polarized_grids,parser_args['npix_l'],parser_args['npix_m'])
    else: raise Exception("Unimplemenented: can only derive ['I','Q','U','V'] from ['RR','RL','LR','LL'], ['XX','XY','YX','YY'] at this time")
      
	
  #now invert, detaper and write out all the facets to disk:  
  if parser_args['facet_centres'] == None:
    dirty = np.abs(fft_utils.ifft2(gridded_vis[0,:,:]))
    dirty_max = np.max(dirty)
    dirty_min = np.min(dirty)
    dirty = ((dirty - dirty_min)*((1-0)/(dirty_max-dirty_min))+0)
    dirty = (dirty / conv._F_detaper).astype(np.float32)
    if parser_args['output_format'] == 'png':
      if parser_args['output_psf']:
	psf = np.real(fft_utils.ifft2(sampling_funct[0,:,:].reshape(parser_args['npix_l'],parser_args['npix_m'])))
	i_psf = pylab.imshow(psf[::-1,:],interpolation='nearest',cmap = pylab.get_cmap('hot'))
	i_psf.write_png(parser_args['output_prefix']+'.psf.png',noscale=True)
	pylab.close('all')
      i = pylab.imshow(dirty[::-1,:],interpolation='nearest',cmap = pylab.get_cmap('hot'),
			 extent=[0, parser_args['npix_l']-1, 0, parser_args['npix_m']-1])
      i.write_png(parser_args['output_prefix']+'.png',noscale=True)
      pylab.close('all')
    else:
      if parser_args['output_psf']:
	psf = np.real(fft_utils.ifft2(sampling_funct[0,:,:].reshape(parser_args['npix_l'],parser_args['npix_m'])))
	fits_export.save_to_fits_image(parser_args['output_prefix']+'.psf.fits',
				       parser_args['npix_l'],parser_args['npix_m'],
				       quantity(parser_args['cell_l'],'arcsec'),quantity(parser_args['cell_m'],'arcsec'),
				       quantity(data._field_centres[parser_args['field_id'],0,0],'arcsec'),
				       quantity(data._field_centres[parser_args['field_id'],0,1],'arcsec'),
				       parser_args['pol'],
				       psf)
      fits_export.save_to_fits_image(parser_args['output_prefix']+'.fits',
				     parser_args['npix_l'],parser_args['npix_m'],
				     quantity(parser_args['cell_l'],'arcsec'),quantity(parser_args['cell_m'],'arcsec'),
				     quantity(data._field_centres[parser_args['field_id'],0,0],'arcsec'),
				     quantity(data._field_centres[parser_args['field_id'],0,1],'arcsec'),
				     parser_args['pol'],
				     dirty)
  else:
    for f in range(0, num_facet_centres):
      dirty = (np.abs(fft_utils.ifft2(gridded_vis[f,:,:]))).reshape(parser_args['npix_l'],parser_args['npix_m'])
      dirty_max = np.max(dirty)
      dirty_min = np.min(dirty)
      dirty = ((dirty - dirty_min)*((1-0)/(dirty_max-dirty_min))+0)
      dirty = (dirty / conv._F_detaper).astype(np.float32)
      if parser_args['output_format'] == 'png':
	if parser_args['output_psf']:
	  psf = np.real(fft_utils.ifft2(sampling_funct[f,:,:].reshape(parser_args['npix_l'],parser_args['npix_m'])))
	  i_psf = pylab.imshow(psf[::-1,:],interpolation='nearest',cmap = pylab.get_cmap('hot'),
		       extent=[0, parser_args['npix_l']-1, 0, parser_args['npix_m']-1])
	  i_psf.write_png(parser_args['output_prefix']+str(f)+'.psf.png',noscale=True)
	  pylab.close('all')
	i = pylab.imshow(dirty[::-1,:],interpolation='nearest',cmap = pylab.get_cmap('hot'),
		       extent=[0, parser_args['npix_l']-1, 0, parser_args['npix_m']-1])
	i.write_png(parser_args['output_prefix']+str(f)+'.png',noscale=True)
	pylab.close('all')
      else:
	if parser_args['output_psf']:
	  psf = np.real(fft_utils.ifft2(sampling_funct[f,:,:].reshape(parser_args['npix_l'],parser_args['npix_m'])))
	  fits_export.save_to_fits_image(parser_args['output_prefix']+str(f)+'.psf.fits',
				       parser_args['npix_l'],parser_args['npix_m'],
				       quantity(parser_args['cell_l'],'arcsec'),quantity(parser_args['cell_m'],'arcsec'),
				       quantity(facet_centres[0,0],'arcsec'),quantity(facet_centres[0,1],'arcsec'),
				       parser_args['pol'],
				       psf)
	fits_export.save_to_fits_image(parser_args['output_prefix']+str(f)+'.fits',
				       parser_args['npix_l'],parser_args['npix_m'],
				       quantity(parser_args['cell_l'],'arcsec'),quantity(parser_args['cell_m'],'arcsec'),
				       quantity(facet_centres[0,0],'arcsec'),quantity(facet_centres[0,1],'arcsec'),
				       parser_args['pol'],
				       dirty)