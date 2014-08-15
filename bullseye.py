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
from helpers import gridding_parameters
from helpers import png_export
import ctypes
libimaging = ctypes.pydll.LoadLibrary("build/algorithms/libimaging.so")
	   
def coords(s):  
    try:
	sT = s.strip()
        ra, dec = map(float, sT[1:len(sT)-1].split(','))
        return ra, dec
    except:
        raise argparse.ArgumentTypeError("Coordinates must be ra,dec tupples")

def measurement_names(s):
    try:
	sT = s.strip()
        ms_names = sT.split(',')
        return 
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
  parser.add_argument('--sample_weighting',help='Specify weighting technique in use.',choices=['natural','uniform'], default='natural')
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
  
  #check how many facets we have to create (if none we'll do normal gridding without any transformations)
  facet_centres = None
  num_facet_centres = 0
  if (parser_args['facet_centres'] != None):
    num_facet_centres = len(parser_args['facet_centres'])
    facet_centres = np.array(parser_args['facet_centres']).astype(base_types.uvw_type)
  if parser_args['do_jones_corrections'] and num_facet_centres != data._cal_no_dirs:
    raise argparse.ArgumentTypeError("Number of calibrated directions does not correspond to number of directions being faceted")
  
  #allocate enough memory to compute image and or facets
  gridded_vis = None
  g = None
  sampling_funct = None
  if pol_options[parser_args['pol']] in data._polarization_correlations.tolist() and not parser_args['do_jones_corrections']:
	num_grids = 1 if (num_facet_centres == 0) else num_facet_centres
	g = np.zeros([num_grids,1,parser_args['npix_l'],parser_args['npix_m']],dtype=base_types.grid_type)
	if parser_args['output_psf'] or parser_args['sample_weighting'] != 'natural':
	  sampling_funct = np.zeros([num_grids,1,parser_args['npix_l'],parser_args['npix_m']],dtype=base_types.psf_type)
  else:
	num_polarized_grids = 1 if (num_facet_centres == 0) else num_facet_centres
	g = np.zeros([num_polarized_grids,4,parser_args['npix_l'],parser_args['npix_m']],dtype=base_types.grid_type)
	if parser_args['output_psf'] or parser_args['sample_weighting'] != 'natural':
	  sampling_funct = np.zeros([num_polarized_grids,1,parser_args['npix_l'],parser_args['npix_m']],dtype=base_types.psf_type) #only one Sampling function over all polarizations
	  
  for chunk_index in range(0,no_chunks):
      #carve up the data in this measurement set:
      chunk_lbound = chunk_index * chunk_size
      chunk_ubound = min((chunk_index+1) * chunk_size,data._no_rows)
      chunk_linecount = chunk_ubound - chunk_lbound
      print "READING CHUNK %d OF %d" % (chunk_index+1,no_chunks)
      data.read_data(start_row=chunk_lbound,no_rows=chunk_linecount,data_column = parser_args['data_column'])
      #fill out the common parameters for gridding:
      params = gridding_parameters.gridding_parameters()
      params.visibilities = data._arr_data.ctypes.data_as(ctypes.c_void_p)
      params.uvw_coords = data._arr_uvw.ctypes.data_as(ctypes.c_void_p)
      params.reference_wavelengths = data._chan_wavelengths.ctypes.data_as(ctypes.c_void_p)
      params.visibility_weights = data._arr_weights.ctypes.data_as(ctypes.c_void_p)
      params.flags = data._arr_flaged.ctypes.data_as(ctypes.c_void_p)
      params.flagged_rows = data._arr_flagged_rows.ctypes.data_as(ctypes.c_void_p)
      params.field_array = data._row_field_id.ctypes.data_as(ctypes.c_void_p)
      params.spw_index_array = data._description_col.ctypes.data_as(ctypes.c_void_p)
      params.imaging_field = ctypes.c_uint(parser_args['field_id'])
      params.baseline_count = ctypes.c_size_t(data._no_baselines)
      params.row_count = ctypes.c_size_t(chunk_linecount)
      params.channel_count = ctypes.c_size_t(data._no_channels)
      params.number_of_polarization_terms = ctypes.c_size_t(data._no_polarization_correlations)
      params.spw_count = ctypes.c_size_t(data._no_spw)
      params.no_timestamps_read = ctypes.c_size_t(data._no_timestamps_read)
      params.nx = ctypes.c_size_t(parser_args['npix_l'])
      params.ny = ctypes.c_size_t(parser_args['npix_m'])
      params.cell_size_x = base_types.uvw_ctypes_convert_type(parser_args['cell_l'])
      params.cell_size_y = base_types.uvw_ctypes_convert_type(parser_args['cell_m'])
      params.conv = conv._conv_FIR.ctypes.data_as(ctypes.c_void_p)
      params.conv_support = ctypes.c_size_t(parser_args['conv_sup'])
      params.conv_oversample = ctypes.c_size_t(parser_args['conv_oversamp'])
      params.phase_centre_ra = base_types.uvw_ctypes_convert_type(data._field_centres[parser_args['field_id'],0,0])
      params.phase_centre_dec = base_types.uvw_ctypes_convert_type(data._field_centres[parser_args['field_id'],0,1])
      params.should_invert_jones_terms = ctypes.c_bool(parser_args['do_jones_corrections'])
      params.antenna_count = ctypes.c_size_t(data._no_antennae)
      params.output_buffer = g.ctypes.data_as(ctypes.c_void_p)
      #no need to grid more than one of the correlations if the user isn't interrested in imaging one of the stokes terms (I,Q,U,V) or the stokes terms are the correlation products:
      if pol_options[parser_args['pol']] in data._polarization_correlations.tolist() and not parser_args['do_jones_corrections']:
	pol_index = data._polarization_correlations.tolist().index(pol_options[parser_args['pol']])
	params.polarization_index = ctypes.c_size_t(pol_index)
	if (parser_args['facet_centres'] == None):
	  if parser_args['output_psf'] or parser_args['sample_weighting'] != 'natural':
	    params.sampling_function_buffer = sampling_funct.ctypes.data_as(ctypes.c_void_p)
	    libimaging.grid_single_pol_with_sampling_func(ctypes.byref(params))
	  else:
	    libimaging.grid_single_pol(ctypes.byref(params))
	else: #facet single correlation term
	  params.facet_centres = facet_centres.ctypes.data_as(ctypes.c_void_p)
	  params.num_facet_centres = ctypes.c_size_t(num_facet_centres)
	  if parser_args['output_psf'] or parser_args['sample_weighting'] != 'natural':
	    params.sampling_function_buffer = sampling_funct.ctypes.data_as(ctypes.c_void_p)
	    libimaging.facet_single_pol_with_sampling_func(ctypes.byref(params))
	  else:
	    libimaging.facet_single_pol(ctypes.byref(params))
	
	if chunk_index == no_chunks - 1:
	  if parser_args['sample_weighting'] == 'uniform':
	    for f in range(0,max(num_facet_centres,1)):
	      g[f,0,:,:] /= (sampling_funct[f,:,:].reshape(parser_args['npix_l'],parser_args['npix_m']) + 0.00000001) #plus some epsilon to make sure we don't get divide by zero issues here
	  gridded_vis = g[:,0,:,:]
      else: #the user want to derive one of the stokes terms (I,Q,U,V) from the correlation terms, or at least want to apply corrective terms
	if (parser_args['facet_centres'] == None): #don't do faceting
	  if parser_args['output_psf'] or parser_args['sample_weighting'] != 'natural':
	    params.sampling_function_buffer = sampling_funct.ctypes.data_as(ctypes.c_void_p)
	    libimaging.grid_4_cor_with_sampling_func(ctypes.byref(params))
	  else:
	    libimaging.grid_4_cor(ctypes.byref(params))
	else:
	  params.facet_centres = facet_centres.ctypes.data_as(ctypes.c_void_p)
	  params.num_facet_centres = ctypes.c_size_t(num_facet_centres)
	  if parser_args['do_jones_corrections']: #do faceting with jones corrections
	    params.jones_terms = data._jones_terms.ctypes.data_as(ctypes.c_void_p)
	    params.antenna_1_ids = data._arr_antenna_1.ctypes.data_as(ctypes.c_void_p)
	    params.antenna_2_ids = data._arr_antenna_2.ctypes.data_as(ctypes.c_void_p)
	    params.timestamp_ids = data._time_indicies.ctypes.data_as(ctypes.c_void_p)
	    if parser_args['output_psf'] or parser_args['sample_weighting'] != 'natural':
	      params.sampling_function_buffer = sampling_funct.ctypes.data_as(ctypes.c_void_p)
	      libimaging.facet_4_cor_corrections_with_sampling_func(ctypes.byref(params))
	    else:
	      libimaging.facet_4_cor_corrections(ctypes.byref(params))
	  else: #skip the jones corrections
	    if parser_args['output_psf'] or parser_args['sample_weighting'] != 'natural':
	      params.sampling_function_buffer = sampling_funct.ctypes.data_as(ctypes.c_void_p)
	      libimaging.facet_4_cor_with_sampling_func(ctypes.byref(params))
	    else:
	      libimaging.facet_4_cor(ctypes.byref(params))
	if chunk_index == no_chunks - 1:      
	  if parser_args['sample_weighting'] == 'uniform':
	    for f in range(0,max(num_facet_centres,1)):
	      for p in range(0,4):
		g[f,p,:,:] /= (sampling_funct[f,:,:].reshape(parser_args['npix_l'],parser_args['npix_m']) + 0.00000001) #plus some epsilon to make sure we don't get divide by zero issues here  
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
	    elif parser_args['pol'] in ['RR','RL','LR','LL']:
	      pol_index = data._polarization_correlations.tolist().index(pol_options[parser_args['pol']])
	      gridded_vis = g[:,pol_index,:,:]
	  elif data._polarization_correlations.tolist() == [pol_options['XX'],pol_options['XY'],pol_options['YX'],pol_options['YY']]:		#linear correlation products
	    if parser_args['pol'] == "I":
	      gridded_vis = ((g[:,0,:,:] + g[:,3,:,:])).reshape(num_polarized_grids,parser_args['npix_l'],parser_args['npix_m'])
	    elif parser_args['pol'] == "V":
	      gridded_vis = ((g[:,1,:,:] - g[:,2,:,:])/1.0).reshape(num_polarized_grids,parser_args['npix_l'],parser_args['npix_m'])
	    elif parser_args['pol'] == "Q":
	      gridded_vis = ((g[:,0,:,:] - g[:,3,:,:])).reshape(num_polarized_grids,parser_args['npix_l'],parser_args['npix_m'])
	    elif parser_args['pol'] == "U":
	      gridded_vis = ((g[:,1,:,:] + g[:,2,:,:])).reshape(num_polarized_grids,parser_args['npix_l'],parser_args['npix_m'])
	    elif parser_args['pol'] in ['XX','XY','YX','YY']:
	      pol_index = data._polarization_correlations.tolist().index(pol_options[parser_args['pol']])
	      gridded_vis = g[:,pol_index,:,:]
	  else:
	    pass #any cases not stated here should be flagged by sanity checks on the program arguement list
  
  #now invert, detaper and write out all the facets to disk:  
  for f in range(0, max(1,num_facet_centres)):
    image_prefix = parser_args['output_prefix'] if num_facet_centres == 0 else parser_args['output_prefix']+".facet"+str(f)
    if parser_args['output_format'] == 'png':
      dirty = (np.real(fft_utils.ifft2(gridded_vis[f,:,:].reshape(parser_args['npix_l'],parser_args['npix_m']))) / conv._F_detaper).astype(np.float32)
      png_export.png_export(dirty,image_prefix,None)
      if parser_args['output_psf']:
	psf = np.real(fft_utils.ifft2(sampling_funct[f,:,:].reshape(parser_args['npix_l'],parser_args['npix_m'])))
	png_export.png_export(psf,image_prefix+'.psf',None)
    else: #export to FITS cube
      dirty = (np.real(fft_utils.ifft2(gridded_vis[f,:,:].reshape(parser_args['npix_l'],parser_args['npix_m']))) / conv._F_detaper).astype(np.float32)
      fits_export.save_to_fits_image(image_prefix+'.fits',
				     parser_args['npix_l'],parser_args['npix_m'],
				     quantity(parser_args['cell_l'],'arcsec'),quantity(parser_args['cell_m'],'arcsec'),
				     quantity(data._field_centres[parser_args['field_id'],0,0],'arcsec'),
				     quantity(data._field_centres[parser_args['field_id'],0,1],'arcsec'),
				     parser_args['pol'],
				     dirty)
      if parser_args['output_psf']:
	psf = np.real(fft_utils.ifft2(sampling_funct[f,:,:].reshape(parser_args['npix_l'],parser_args['npix_m'])))
	fits_export.save_to_fits_image(image_prefix+'.psf.fits',
				       parser_args['npix_l'],parser_args['npix_m'],
				       quantity(parser_args['cell_l'],'arcsec'),quantity(parser_args['cell_m'],'arcsec'),
				       quantity(data._field_centres[parser_args['field_id'],0,0],'arcsec'),
				       quantity(data._field_centres[parser_args['field_id'],0,1],'arcsec'),
				       parser_args['pol'],
				       psf)
  print "TERMINATED SUCCESSFULLY"