#!/usr/bin/python
import sys
import argparse
from pyparsing import commaSeparatedList
import numpy as np
import pylab
import re
from pyrap.quanta import quantity
import concurrent.futures as cf

from helpers import data_set_loader
from helpers import fft_utils
from helpers import convolution_filter
from helpers import fits_export
from helpers import base_types
from helpers import gridding_parameters
from helpers import png_export
from helpers import timer
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
  io = cf.ThreadPoolExecutor(1)
  
  total_run_time = timer.timer()
  total_run_time.start()
  inversion_timer = timer.timer()
  filter_creation_timer = timer.timer()
  parser = argparse.ArgumentParser(description='Bullseye: An implementation of targetted facet-based synthesis imaging in radio astronomy.')
  pol_options = {'I' : 1, 'Q' : 2, 'U' : 3, 'V' : 4, 'RR' : 5, 'RL' : 6, 'LR' : 7, 'LL' : 8, 'XX' : 9, 'XY' : 10, 'YX' : 11, 'YY' : 12} # as per Stokes.h in casacore, the rest is left unimplemented
  '''
    See Smirnov I (2011) for description on conversion between correlation terms and stokes params for linearly polarized feeds
    See Synthesis Imaging II (1999) Pg. 9 for a description on conversion between correlation terms and stokes params for circularly polarized feeds
  '''
  pol_dependencies = {
    'I'  : [[pol_options['I']],[pol_options['RR'],pol_options['LL']],[pol_options['XX'],pol_options['YY']]],
    'V'  : [[pol_options['V']],[pol_options['RR'],pol_options['LL']],[pol_options['XY'],pol_options['YX']]],
    'U'  : [[pol_options['U']],[pol_options['RL'],pol_options['LR']],[pol_options['XY'],pol_options['YX']]],
    'Q'  : [[pol_options['Q']],[pol_options['RL'],pol_options['LR']],[pol_options['XX'],pol_options['YY']]],
    'RR' : [[pol_options['RR']]],
    'RL' : [[pol_options['RL']]],
    'LR' : [[pol_options['LR']]],
    'LL' : [[pol_options['LL']]],
    'XX' : [[pol_options['XX']]],
    'XY' : [[pol_options['XY']]],
    'YX' : [[pol_options['YX']]],
    'YY' : [[pol_options['YY']]]
  }
  feed_descriptions = {
    'I'  : ["stokes","circular","linear"],
    'V'  : ["stokes","circular","linear"],
    'U'  : ["stokes","circular","linear"],
    'Q'  : ["stokes","circular","linear"],
    'RR' : ["circular"],
    'RL' : ["circular"],
    'LR' : ["circular"],
    'LL' : ["circular"],
    'XX' : ["linear"],
    'XY' : ["linear"],
    'YX' : ["linear"],
    'YY' : ["linear"]
  }
  parser.add_argument('input_ms', help='Name of the measurement set(s) to read. Multiple MSs must be comma-delimited without any separating spaces, eg. "\'one.ms\',\'two.ms\'"', type=str)
  parser.add_argument('--output_prefix', help='Prefix for the output FITS images. Facets will be indexed as [prefix_1.fits ... prefix_n.fits]', type=str, default='out.bullseye')
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
  '''
  TODO: FIX
  parser.add_argument('--output_psf',help='Outputs the point-spread-function',type=bool,default=False)
  parser.add_argument('--sample_weighting',help='Specify weighting technique in use.',choices=['natural','uniform'], default='natural')
  '''
  parser_args = vars(parser.parse_args())
  
  #initially the output grids must be set to NONE. Memory will only be allocated before the first MS is read.
  gridded_vis = None
  sampling_funct = None
  ms_names = commaSeparatedList.parseString(parser_args['input_ms'])
  for ms_index,ms in enumerate(ms_names):
    print "NOW IMAGING %s" % ms
    data = data_set_loader.data_set_loader(ms,read_jones_terms=parser_args['do_jones_corrections'])
    data.read_head()
    chunk_size = data.compute_number_of_rows_to_read_from_mem_requirements(parser_args['mem_available_for_input_data'])
    if chunk_size == 0:
      raise Exception("Insufficient memory allocated for loading data. Cannot even load a single row and a timestamp of jones matricies at a time")
    no_chunks = data.number_of_read_iterations_required_from_mem_requirements(parser_args['mem_available_for_input_data'])
    #some sanity checks:
    #check that the measurement set contains the correlation terms necessary to create the requested pollarization / Stokes term:
    correlations_to_grid = None
    feeds_in_use = None
    for feed_index,req in enumerate(pol_dependencies[parser_args['pol']]):
      if set(req).issubset(data._polarization_correlations.tolist()):
	correlations_to_grid = req #we found a subset that must be gridded
	feeds_in_use = feed_descriptions[parser_args['pol']][feed_index]
    if correlations_to_grid == None:
      raise argparse.ArgumentTypeError("Cannot obtain requested gridded polarization from the provided measurement set. Need one of %s" % pol_dependencies[parser_args['pol']])
    #check that we have 4 correlations before trying to apply jones corrective terms:
    if parser_args['do_jones_corrections'] and data._polarization_correlations.tolist() not in [
						       [pol_options['RR'],pol_options['RL'],pol_options['LR'],pol_options['LL']],	#4 circular correlation products
						       [pol_options['XX'],pol_options['XY'],pol_options['YX'],pol_options['YY']],	#4 linear correlation products
						       [pol_options['I'],pol_options['Q'],pol_options['U'],pol_options['V']]		#4 stokes terms
						       ]:	
      raise argparse.ArgumentTypeError("Applying jones corrective terms require a measurement set with 4 linear or 4 circular or 4 stokes terms")
    if parser_args['do_jones_corrections'] and (not data._dde_cal_info_exists or not data._dde_cal_info_desc_exists):
      raise argparse.ArgumentTypeError("Measurement set does not contain corrective DDE terms or the description table is missing.")
    
    if parser_args['field_id'] not in range(0,len(data._field_centre_names)):
      raise argparse.ArgumentTypeError("Specified field does not exist Must be in 0 ... %d for this Measurement Set" % (len(data._field_centre_names) - 1))
    with filter_creation_timer:
      conv = convolution_filter.convolution_filter(parser_args['conv_sup'],
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
  
    #allocate enough memory to compute image and or facets (only before gridding the first MS)
    if not parser_args['do_jones_corrections']:
	if gridded_vis == None:
	  num_grids = 1 if (num_facet_centres == 0) else num_facet_centres
	  gridded_vis = np.zeros([num_grids,len(correlations_to_grid),parser_args['npix_l'],parser_args['npix_m']],dtype=base_types.grid_type)
    else:
	if gridded_vis == None:
	  num_polarized_grids = 1 if (num_facet_centres == 0) else num_facet_centres
	  gridded_vis = np.zeros([num_polarized_grids,4,parser_args['npix_l'],parser_args['npix_m']],dtype=base_types.grid_type)
    '''
    TODO: FIX
    if sampling_funct == None:
	  if parser_args['output_psf'] or parser_args['sample_weighting'] != 'natural':
	    num_grids = 1 if (num_facet_centres == 0) else num_facet_centres
	    sampling_funct = np.zeros([num_grids,1,parser_args['npix_l'],parser_args['npix_m']],dtype=base_types.psf_type)
    '''
    
    '''
    each chunk will start processing while data is being read in, we will wait until until this process rejoins
    before copying the buffers and starting to grid the next chunk
    '''
    for chunk_index in range(0,no_chunks):
      #carve up the data in this measurement set:
      chunk_lbound = chunk_index * chunk_size
      chunk_ubound = min((chunk_index+1) * chunk_size,data._no_rows)
      chunk_linecount = chunk_ubound - chunk_lbound
      print "READING CHUNK %d OF %d" % (chunk_index+1,no_chunks)
      data.read_data(start_row=chunk_lbound,no_rows=chunk_linecount,data_column = parser_args['data_column'])
      #after the compute of the previous cycle finishes make deep copies and 
      #fill out the common parameters for gridding this cycle:
      libimaging.gridding_barrier()   #WARNING: This async barrier is cricical for valid gridding results
     
      params = gridding_parameters.gridding_parameters()
      arr_data_cpy = data._arr_data #gridding will operate on deep copied memory
      params.visibilities = arr_data_cpy.ctypes.data_as(ctypes.c_void_p)
      arr_uvw_cpy = data._arr_uvw #gridding will operate on deep copied memory
      params.uvw_coords = arr_uvw_cpy.ctypes.data_as(ctypes.c_void_p)
      params.reference_wavelengths = data._chan_wavelengths.ctypes.data_as(ctypes.c_void_p) #this is part of the header of the MS and must stay constant between chunks
      arr_weights_cpy = data._arr_weights #gridding will operate on deep copied memory
      params.visibility_weights = arr_weights_cpy.ctypes.data_as(ctypes.c_void_p)
      arr_flags_cpy = data._arr_flaged #gridding will operate on deep copied memory
      params.flags = arr_flags_cpy.ctypes.data_as(ctypes.c_void_p)
      arr_flagged_rows_cpy = data._arr_flagged_rows #gridding will operate on deep copied memory
      params.flagged_rows = arr_flagged_rows_cpy.ctypes.data_as(ctypes.c_void_p)
      row_field_id_cpy = data._row_field_id #gridding will operate on deep copied memory
      params.field_array = row_field_id_cpy.ctypes.data_as(ctypes.c_void_p)
      arr_description_col_cpy = data._description_col #gridding will operate on deep copied memory
      params.spw_index_array = arr_description_col_cpy.ctypes.data_as(ctypes.c_void_p)
      params.imaging_field = ctypes.c_uint(parser_args['field_id']) #this ensures a deep copy
      params.baseline_count = ctypes.c_size_t(data._no_baselines) #this ensures a deep copy
      params.row_count = ctypes.c_size_t(chunk_linecount) #this ensures a deep copy
      params.channel_count = ctypes.c_size_t(data._no_channels) #this ensures a deep copy
      params.number_of_polarization_terms = ctypes.c_size_t(data._no_polarization_correlations) #this ensures a deep copy
      params.spw_count = ctypes.c_size_t(data._no_spw) #this ensures a deep copy
      params.no_timestamps_read = ctypes.c_size_t(data._no_timestamps_read) #this ensures a deep copy
      params.nx = ctypes.c_size_t(parser_args['npix_l']) #this ensures a deep copy
      params.ny = ctypes.c_size_t(parser_args['npix_m']) #this ensures a deep copy
      params.cell_size_x = base_types.uvw_ctypes_convert_type(parser_args['cell_l']) #this ensures a deep copy
      params.cell_size_y = base_types.uvw_ctypes_convert_type(parser_args['cell_m']) #this ensures a deep copy
      params.conv = conv._conv_FIR.ctypes.data_as(ctypes.c_void_p) #this won't change between chunks
      params.conv_support = ctypes.c_size_t(parser_args['conv_sup']) #this ensures a deep copy
      params.conv_oversample = ctypes.c_size_t(parser_args['conv_oversamp'])#this ensures a deep copy
      params.phase_centre_ra = base_types.uvw_ctypes_convert_type(data._field_centres[parser_args['field_id'],0,0]) #this ensures a deep copy
      params.phase_centre_dec = base_types.uvw_ctypes_convert_type(data._field_centres[parser_args['field_id'],0,1]) #this ensures a deep copy
      params.should_invert_jones_terms = ctypes.c_bool(parser_args['do_jones_corrections']) #this ensures a deep copy
      params.antenna_count = ctypes.c_size_t(data._no_antennae) #this ensures a deep copy
      params.output_buffer = gridded_vis.ctypes.data_as(ctypes.c_void_p) #we never do 2 computes at the same time (or the reduction is handled at the C++ implementation level)
      
      #no need to grid more than one of the correlations if the user isn't interrested in imaging one of the stokes terms (I,Q,U,V) or the stokes terms are the correlation products:
      if len(correlations_to_grid) == 1 and not parser_args['do_jones_corrections']:
	pol_index = data._polarization_correlations.tolist().index(correlations_to_grid[0])
	params.polarization_index = ctypes.c_size_t(pol_index)
	if (parser_args['facet_centres'] == None):
	  libimaging.grid_single_pol(ctypes.byref(params))
	else: #facet single correlation term
	  params.facet_centres = facet_centres.ctypes.data_as(ctypes.c_void_p)
	  params.num_facet_centres = ctypes.c_size_t(num_facet_centres)
	  libimaging.facet_single_pol(ctypes.byref(params))
      elif len(correlations_to_grid) == 2 and not parser_args['do_jones_corrections']: #the user want to derive one of the stokes terms (I,Q,U,V)
	pol_index = data._polarization_correlations.tolist().index(correlations_to_grid[0])
	pol_index_2 = data._polarization_correlations.tolist().index(correlations_to_grid[1])
	params.polarization_index = ctypes.c_size_t(pol_index)
	params.second_polarization_index = ctypes.c_size_t(pol_index_2)
	if (parser_args['facet_centres'] == None):
	  libimaging.grid_single_pol(ctypes.byref(params))
	else: #facet single correlation term
	  params.facet_centres = facet_centres.ctypes.data_as(ctypes.c_void_p)
	  params.num_facet_centres = ctypes.c_size_t(num_facet_centres)
	  libimaging.facet_single_pol(ctypes.byref(params))
      else: #the user want to apply corrective terms
	if (parser_args['facet_centres'] == None): #don't do faceting
	  libimaging.grid_4_cor(ctypes.byref(params))
	else:
	  params.facet_centres = facet_centres.ctypes.data_as(ctypes.c_void_p)
	  params.num_facet_centres = ctypes.c_size_t(num_facet_centres)
	  if parser_args['do_jones_corrections']: #do faceting with jones corrections
	    params.jones_terms = data._jones_terms.ctypes.data_as(ctypes.c_void_p)
	    params.antenna_1_ids = data._arr_antenna_1.ctypes.data_as(ctypes.c_void_p)
	    params.antenna_2_ids = data._arr_antenna_2.ctypes.data_as(ctypes.c_void_p)
	    params.timestamp_ids = data._time_indicies.ctypes.data_as(ctypes.c_void_p)
	    libimaging.facet_4_cor_corrections(ctypes.byref(params))
	  else: #skip the jones corrections
	    libimaging.facet_4_cor(ctypes.byref(params))
     
      if chunk_index == no_chunks - 1:
	libimaging.gridding_barrier()
      
  '''
  TODO: FIX THESE
  if parser_args['output_psf'] or parser_args['sample_weighting'] != 'natural':
    pass #TODO:implement a way to compute the PSF
  if parser_args['sample_weighting'] != 'natural':
    pass #TODO:implement uniform weighting
  '''
  
  '''
  See Smirnov I (2011) for description on conversion between correlation terms and stokes params for linearly polarized feeds
  See Synthesis Imaging II (1999) Pg. 9 for a description on conversion between correlation terms and stokes params for circularly polarized feeds
  '''
  if parser_args['do_jones_corrections']:
    if feeds_in_use == 'circular':		#circular correlation products
      if parser_args['pol'] == "I":
	IpV = data._polarization_correlations.tolist().index(pol_options['RR'])
	ImV = data._polarization_correlations.tolist().index(pol_options['LL'])
	gridded_vis[:,0,:,:] = ((gridded_vis[:,IpV,:,:] + gridded_vis[:,ImV,:,:])/2)
      elif parser_args['pol'] == "V":
	IpV = data._polarization_correlations.tolist().index(pol_options['RR'])
	ImV = data._polarization_correlations.tolist().index(pol_options['LL'])
	gridded_vis[:,0,:,:] = ((gridded_vis[:,IpV,:,:] - gridded_vis[:,ImV,:,:])/2)
      elif parser_args['pol'] == "Q":
	QpiU = data._polarization_correlations.tolist().index(pol_options['RL'])
	QmiU = data._polarization_correlations.tolist().index(pol_options['LR'])
	gridded_vis[:,0,:,:] = ((gridded_vis[:,QpiU,:,:] + gridded_vis[:,QmiU,:,:])/2)
      elif parser_args['pol'] == "U":
	QpiU = data._polarization_correlations.tolist().index(pol_options['RL'])
	QmiU = data._polarization_correlations.tolist().index(pol_options['LR'])
	gridded_vis[:,0,:,:] = ((gridded_vis[:,QpiU,:,:] - gridded_vis[:,QmiU,:,:])/2.0)
      elif parser_args['pol'] in ['RR','RL','LR','LL']:
	pass #already stored in gridded_vis[:,0,:,:]
    elif feeds_in_use == 'linear':		#linear correlation products
      if parser_args['pol'] == "I":
	IpQ = data._polarization_correlations.tolist().index(pol_options['XX'])
	ImQ = data._polarization_correlations.tolist().index(pol_options['YY'])
	gridded_vis[:,0,:,:] = ((gridded_vis[:,IpQ,:,:] + gridded_vis[:,ImQ,:,:]))
      elif parser_args['pol'] == "Q":
	IpQ = data._polarization_correlations.tolist().index(pol_options['XX'])
	ImQ = data._polarization_correlations.tolist().index(pol_options['YY'])
	gridded_vis[:,0,:,:] = ((gridded_vis[:,IpQ,:,:] - gridded_vis[:,ImQ,:,:]))
      elif parser_args['pol'] == "U":
	UpiV = data._polarization_correlations.tolist().index(pol_options['XY'])
	UmiV = data._polarization_correlations.tolist().index(pol_options['YX'])
	gridded_vis[:,0,:,:] = ((gridded_vis[:,UpiV,:,:] + gridded_vis[:,UmiV,:,:]))
      elif parser_args['pol'] == "V":
	UpiV = data._polarization_correlations.tolist().index(pol_options['XY'])
	UmiV = data._polarization_correlations.tolist().index(pol_options['YX'])
	gridded_vis[:,0,:,:] = ((gridded_vis[:,UpiV,:,:] - gridded_vis[:,UmiV,:,:]))/1.0
      elif parser_args['pol'] in ['XX','XY','YX','YY']:
	pass #already stored in gridded_vis[:,0,:,:]
    else:
      pass #any cases not stated here should be flagged by sanity checks on the program arguement list
  else:
    if feeds_in_use == 'circular':		#circular correlation products
      if parser_args['pol'] == "I":
	IpV = correlations_to_grid.index(pol_options['RR'])
	ImV = correlations_to_grid.index(pol_options['LL'])
	gridded_vis[:,0,:,:] = ((gridded_vis[:,IpV,:,:] + gridded_vis[:,ImV,:,:])/2)
      elif parser_args['pol'] == "V":
	IpV = correlations_to_grid.index(pol_options['RR'])
	ImV = correlations_to_grid.index(pol_options['LL'])
	gridded_vis[:,0,:,:] = ((gridded_vis[:,IpV,:,:] - gridded_vis[:,ImV,:,:])/2)
      elif parser_args['pol'] == "Q":
	QpiU = correlations_to_grid.index(pol_options['RL'])
	QmiU = correlations_to_grid.index(pol_options['LR'])
	gridded_vis[:,0,:,:] = ((gridded_vis[:,QpiU,:,:] + gridded_vis[:,QmiU,:,:])/2)
      elif parser_args['pol'] == "U":
	QpiU = correlations_to_grid.index(pol_options['RL'])
	QmiU = correlations_to_grid.index(pol_options['LR'])
	gridded_vis[:,0,:,:] = ((gridded_vis[:,QpiU,:,:] - gridded_vis[:,QmiU,:,:])/2)
      elif parser_args['pol'] in ['RR','RL','LR','LL']:
	pass #already stored in gridded_vis[:,0,:,:]
    elif feeds_in_use == 'linear':		#linear correlation products
      if parser_args['pol'] == "I":
	IpQ = correlations_to_grid.index(pol_options['XX'])
	ImQ = correlations_to_grid.index(pol_options['YY'])
	gridded_vis[:,0,:,:] = ((gridded_vis[:,IpQ,:,:] + gridded_vis[:,ImQ,:,:]))
      elif parser_args['pol'] == "Q":
	IpQ = correlations_to_grid.index(pol_options['XX'])
	ImQ = correlations_to_grid.index(pol_options['YY'])
	gridded_vis[:,0,:,:] = ((gridded_vis[:,IpQ,:,:] - gridded_vis[:,ImQ,:,:]))
      elif parser_args['pol'] == "U":
	UpiV = correlations_to_grid.index(pol_options['XY'])
	UmiV = correlations_to_grid.index(pol_options['YX'])
	gridded_vis[:,0,:,:] = ((gridded_vis[:,UpiV,:,:] + gridded_vis[:,UmiV,:,:]))
      elif parser_args['pol'] == "V":
	UpiV = correlations_to_grid.index(pol_options['XY'])
	UmiV = correlations_to_grid.index(pol_options['YX'])
	gridded_vis[:,0,:,:] = ((gridded_vis[:,UpiV,:,:] - gridded_vis[:,UmiV,:,:]))
      elif parser_args['pol'] in ['XX','XY','YX','YY']:
	pass #already stored in gridded_vis[:,0,:,:]
    else:
      pass #any cases not stated here should be flagged by sanity checks on the program arguement list
  #now invert, detaper and write out all the facets to disk:  
  
  for f in range(0, max(1,num_facet_centres)):
    image_prefix = parser_args['output_prefix'] if num_facet_centres == 0 else parser_args['output_prefix']+".facet"+str(f)
    if parser_args['output_format'] == 'png':
      with inversion_timer:
	dirty = (np.real(fft_utils.ifft2(gridded_vis[f,0,:,:].reshape(parser_args['npix_l'],parser_args['npix_m']))) / conv._F_detaper).astype(np.float32)
      png_export.png_export(dirty,image_prefix,None)
      '''
      TODO: FIX
      if parser_args['output_psf']:
	psf = np.real(fft_utils.ifft2(sampling_funct[f,:,:].reshape(parser_args['npix_l'],parser_args['npix_m'])))
	png_export.png_export(psf,image_prefix+'.psf',None)
      '''
    else: #export to FITS cube
      with inversion_timer:
	dirty = (np.real(fft_utils.ifft2(gridded_vis[f,0,:,:].reshape(parser_args['npix_l'],parser_args['npix_m']))) / conv._F_detaper).astype(np.float32)
      fits_export.save_to_fits_image(image_prefix+'.fits',
				     parser_args['npix_l'],parser_args['npix_m'],
				     quantity(parser_args['cell_l'],'arcsec'),quantity(parser_args['cell_m'],'arcsec'),
				     quantity(data._field_centres[parser_args['field_id'],0,0],'arcsec'),
				     quantity(data._field_centres[parser_args['field_id'],0,1],'arcsec'),
				     parser_args['pol'],
				     dirty)
      '''
      TODO: FIX
      if parser_args['output_psf']:
	psf = np.real(fft_utils.ifft2(sampling_funct[f,:,:].reshape(parser_args['npix_l'],parser_args['npix_m'])))
	fits_export.save_to_fits_image(image_prefix+'.psf.fits',
				       parser_args['npix_l'],parser_args['npix_m'],
				       quantity(parser_args['cell_l'],'arcsec'),quantity(parser_args['cell_m'],'arcsec'),
				       quantity(data._field_centres[parser_args['field_id'],0,0],'arcsec'),
				       quantity(data._field_centres[parser_args['field_id'],0,1],'arcsec'),
				       parser_args['pol'],
				       psf)
      '''
  print "FINISHED WORK SUCCESSFULLY"
  total_run_time.stop()
  print "STATISTICS:"
  print "\tConvolution filter and detapering function creation time: %f secs" % filter_creation_timer.elapsed()
  print "\t[In Parallel] Data loading and conversion time: %f secs" % data_set_loader.data_set_loader.time_to_load_chunks.elapsed()
  libimaging.get_gridding_walltime.restype = ctypes.c_double
  print "\t[In Parallel] Gridding time: %f secs" % libimaging.get_gridding_walltime()
  print "\tFourier inversion time: %f secs" % inversion_timer.elapsed()
  print "\tTotal runtime: %f secs" % total_run_time.elapsed()