'''
Bullseye:
An accelerated targeted facet imager
Category: Radio Astronomy / Widefield synthesis imaging

Authors: Benjamin Hugo, Oleg Smirnov, Cyril Tasse, James Gain
Contact: hgxben001@myuct.ac.za

Copyright (C) 2014-2015 Rhodes Centre for Radio Astronomy Techniques and Technologies
Department of Physics and Electronics
Rhodes University
Artillery Road P O Box 94
Grahamstown
6140
Eastern Cape South Africa

Copyright (C) 2014-2015 Department of Computer Science
University of Cape Town
18 University Avenue
University of Cape Town
Rondebosch
Cape Town
South Africa

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
'''

#!/usr/bin/python
import argparse
import numpy as np
from pyparsing import commaSeparatedList
import pylab
import os

from pyrap.quanta import quantity
import ctypes

import helpers.channel_indexer as channel_indexer
import helpers.command_line_options as command_line_options
import helpers.facet_list_parser as facet_list_parser
import helpers.stokes as stokes
import bullseye_mo.library_loader as library_loader
from helpers import timer
from helpers import png_export
from helpers import fits_export 
if __name__ == "__main__":
  total_run_time = timer.timer()
  total_run_time.start()

  '''
  Parse command line arguments
  '''
  (parser,parser_args) = command_line_options.build_command_line_options_parser()

  '''
  Pick a backend to use
  '''
  libimaging = library_loader.load_library(parser_args['use_back_end'],parser_args['precision'])
  from bullseye_mo import base_types
  base_types.force_precision(parser_args['precision'])
  from helpers import data_set_loader
  from bullseye_mo import convolution_filter
  from bullseye_mo import gridding_parameters
 
  '''
  initially the output grids must be set to NONE. Memory will only be allocated before the first MS is read.
  '''
  gridded_vis = None
  sampling_funct = None
  ms_names = commaSeparatedList.parseString(parser_args['input_ms'])

  '''
  General strategy for IO and processing:
    for all measurement sets:
      read ms header data (the headers better correspond between measurement sets otherwise the user is really doing something weird and wonderful - this is a primary assumption)
      parse and create a list of facet centres
      parse and create a list of enabled channel ids
      parse and create a list of whereto each of the enabled channels should be gridded (we might do continuim imaging or smear only certain bands together for spectral imaging)
      allocate some memory
      initialize backend infrastructure with data that doesn't change from stride to stride
      now carve up the current measurement set primary table into no_chunks and process each such stride in the following loop:
      for all chunks:
	load chunk from disk (buffer the io)
	wait for any previous gridding operations to finish up
	deep copy chunk
	call the correct gridding method (single / duel / quad with corrections) to operate on the
	in parallel start up the gridder
	if last chunk then normalize else continue on to start buffering the next chunk
      #end for chunks
    #end for ms
    compact and finalize grids
    write out to disk (either png or FITS)
    optionally stitch together using montage
  '''
  for ms_index,ms in enumerate(ms_names):
    print "NOW IMAGING %s" % ms
    data = data_set_loader.data_set_loader(ms,read_jones_terms=parser_args['do_jones_corrections'])
    data.read_head()
    
    '''
    Create convolution filter
    '''
    filter_creation_timer = timer.timer()
    lambda_min = np.min(data._chan_wavelengths)
    #as per Synthesis Imaging II, pg 24. w is maximum at low elevations and when baseline and source vectors have same azimuth (ie. parallel):
    w_max = data._maximum_baseline_length / lambda_min
    print "The maximum w (measured in wavelengths) is estimated to be", w_max  
    with filter_creation_timer:
      conv = convolution_filter.convolution_filter(parser_args['conv_sup'],
						   parser_args['conv_oversamp'],parser_args['conv'],
						   "1D_AA" if parser_args['wplanes'] <= 1 else ("2D_WPROJ" if parser_args['use_back_end'] == 'CPU' else "1D_WPROJ"),parser_args['wplanes'],
						   parser_args['npix_l'],parser_args['npix_m'],
						   parser_args['cell_l'],parser_args['cell_m'],
						   w_max,data._field_centres[parser_args['field_id'],0,0],data._field_centres[parser_args['field_id'],0,1])
    
    no_chunks = parser_args['no_chunks']
    if no_chunks < 1:
      raise Exception("Cannot read less than one chunk from measurement set!")
    chunk_size = int(np.ceil(data._no_rows / float(no_chunks)))
    
    '''
    check that the measurement set contains the correlation terms necessary to create the requested pollarization / Stokes term:
    '''
    (correlations_to_grid, feeds_in_use) = command_line_options.find_necessary_correlations_indexes(parser_args['do_jones_corrections'],parser_args['pol'],data)
    if correlations_to_grid == None:
      raise argparse.ArgumentTypeError("Cannot obtain requested gridded polarization from the provided measurement set. Need one of %s" % command_line_options.pol_dependencies[parser_args['pol']])

    '''
    check that we have 4 correlations before trying to apply jones corrective terms:
    '''
    if (parser_args['do_jones_corrections'] and
	data._polarization_correlations.tolist() not in [
							 #4 circular correlation products:
							 [stokes.pol_options['RR'],stokes.pol_options['RL'],stokes.pol_options['LR'],stokes.pol_options['LL']],
							 #4 linear correlation products:
							 [stokes.pol_options['XX'],stokes.pol_options['XY'],stokes.pol_options['YX'],stokes.pol_options['YY']],
							 #4 stokes terms:
							 [stokes.pol_options['I'],stokes.pol_options['Q'],stokes.pol_options['U'],stokes.pol_options['V']]
							]
      ):
      raise argparse.ArgumentTypeError("Applying jones corrective terms require a measurement set with 4 linear or 4 circular or 4 stokes terms")
    if parser_args['do_jones_corrections'] and (not data._dde_cal_info_exists or not data._dde_cal_info_desc_exists):
      raise argparse.ArgumentTypeError("Measurement set does not contain corrective DDE terms or the description table is missing.")

    '''
    check that the field id is valid
    '''
    if parser_args['field_id'] not in range(0,len(data._field_centre_names)):
      raise argparse.ArgumentTypeError("Specified field does not exist Must be in 0 ... %d for this Measurement Set" % (len(data._field_centre_names) - 1))
    print "IMAGING ONLY FIELD %s" % data._field_centre_names[parser_args['field_id']]

    '''
    check how many facets we have to create (if none we'll do normal gridding without any transformations)
    '''
    num_facet_centres = facet_list_parser.compute_number_of_facet_centres(parser_args)
    if parser_args['stitch_facets']:
      if num_facet_centres < 2:
	raise argparse.ArgumentTypeError("Need at least two facets to perform stitching")
      if parser_args['output_format'] != 'fits':
	raise argparse.ArgumentTypeError("Facet output format must be of type FITS")

    facet_centres = facet_list_parser.create_facet_centre_list(parser_args,data,num_facet_centres)
    facet_list_parser.print_facet_centre_list(facet_centres,num_facet_centres)

    if parser_args['do_jones_corrections'] and num_facet_centres != data._cal_no_dirs:
      raise argparse.ArgumentTypeError("Number of calibrated directions does not correspond to number of directions being faceted")
    '''
    check that the convolution kernel size is not greater than the grid size
    '''
    if (parser_args['conv_sup']*2 + 1) >= min(parser_args['npix_l'],parser_args['npix_m']):
      raise argparse.ArgumentTypeError("Full convolution support must be smaller than the grid size")
    '''
    populate the channels to be imaged:
    '''
    (channels_to_image,enabled_channels) = channel_indexer.parse_channels_to_be_imaged(parser_args['channel_select'],data)

    if channels_to_image[len(channels_to_image)-1] >= data._no_spw*data._no_channels:
      raise argparse.ArgumentTypeError("One or more channels don't exist. Only %d spw's of %d channels each are available" % (data._no_spw,data._no_channels))
    channel_indexer.print_requested_channels(channels_to_image,data)

    '''
    Compute the fits cube reference wavelength and delta wavelength, and check that the
    channels / spw-centres are evenly spaced
    '''
    cube_first_wavelength = 0
    cube_delta_wavelength = 0
    if parser_args['output_format'] == "png" and not parser_args['average_all']:
      raise argparse.ArgumentTypeError("Cannot output cube in png format. Try averaging all channels together (--average_all), or use FITS format")
    if parser_args['average_spw_channels'] and parser_args['average_all']:
      raise argparse.ArgumentTypeError("Both --average_spw_channels and --average_all are enabled. These are mutually exclusive")
    elif len(channels_to_image) > 1 and not (parser_args['average_spw_channels'] or parser_args['average_all']):
      (cube_delta_wavelength,cube_first_wavelength) = channel_indexer.compute_cube_chan_dim_spacing_no_averaging(data,channels_to_image)
    elif len(channels_to_image) > 1 and parser_args['average_spw_channels']:
      (cube_delta_wavelength,cube_first_wavelength) = channel_indexer.compute_cube_chan_dim_spw_averaging(data,channels_to_image)
    elif parser_args['average_all']:
      (cube_delta_wavelength,cube_first_wavelength) = channel_indexer.compute_cube_chan_dim_all_channel_averaging(data)
    else: #only one channel
      (cube_delta_wavelength,cube_first_wavelength) = channel_indexer.compute_cube_chan_dim_single_channel(data,channels_to_image)

    '''
    work out how many channels / averaged SPWs this cube will have (remember each SPW may have 0 <= x <= no_channels enabled)
    '''
    (channel_grid_index,cube_chan_dim_size) = channel_indexer.compute_vis_grid_indicies(parser_args['average_spw_channels'],
									parser_args['average_all'],
									data,
									enabled_channels,
									channels_to_image)
    '''
    Work out to which grid each sampling function (per channel) should be gridded.
    '''
    sampling_function_channel_grid_index = None
    sampling_function_channel_count = 0
    if parser_args['output_psf'] or (parser_args['sample_weighting'] == 'uniform'):
      sampling_function_channel_grid_index,sampling_function_channel_count = channel_indexer.compute_sampling_function_grid_indicies(data,channels_to_image,enabled_channels)
    '''
    Work out how many (pixels) to pad the images with. Filtering normally doesn't cut
    off aliases exactly at the border of the image - we expect some rolloff. Padding helps
    put this rolloff slightly outside the image.
    '''
    padding_per_edge_m = int(np.ceil(parser_args['npix_m'] * (-1.0+parser_args['image_padding']) * 0.5))
    padding_per_edge_l = int(np.ceil(parser_args['npix_l'] * (-1.0+parser_args['image_padding']) * 0.5))
    npix_m = parser_args['npix_m'] + padding_per_edge_m * 2
    npix_l = parser_args['npix_l'] + padding_per_edge_l * 2
    m_left_margin = padding_per_edge_l
    m_right_margin = parser_args['npix_m'] + padding_per_edge_m
    l_left_margin = padding_per_edge_l
    l_right_margin = parser_args['npix_l'] + padding_per_edge_l
    '''
    allocate enough memory to compute image and or facets (only before gridding the first MS)
    '''
    num_facet_grids = 1 if (num_facet_centres == 0) else num_facet_centres
    if not parser_args['do_jones_corrections']:
	if gridded_vis == None:
	  gridded_vis = np.zeros([num_facet_grids,cube_chan_dim_size,len(correlations_to_grid),npix_l,npix_m],dtype=base_types.grid_type)
    else:
	if gridded_vis == None:
	  gridded_vis = np.zeros([num_facet_grids,cube_chan_dim_size,4,npix_l,npix_m],dtype=base_types.grid_type)

    if parser_args['output_psf'] or (parser_args['sample_weighting'] == 'uniform'):
      if sampling_funct == None:
	sampling_funct = np.zeros([num_facet_grids,sampling_function_channel_count,1,npix_l,npix_m],dtype=base_types.psf_type)

    '''
    initiate the backend imaging library
    '''
    params = gridding_parameters.gridding_parameters()
    params.chunk_max_row_count = ctypes.c_size_t(chunk_size)
    params.nx = ctypes.c_size_t(npix_m) #this ensures a deep copy
    params.ny = ctypes.c_size_t(npix_l) #this ensures a deep copy
    params.cell_size_x = base_types.uvw_ctypes_convert_type(parser_args['cell_m']) #this ensures a deep copy
    params.cell_size_y = base_types.uvw_ctypes_convert_type(parser_args['cell_l']) #this ensures a deep copy
    params.conv = conv._conv_FIR.ctypes.data_as(ctypes.c_void_p) #this won't change between chunks
    params.conv_support = ctypes.c_size_t(parser_args['conv_sup']) #this ensures a deep copy
    params.conv_oversample = ctypes.c_size_t(parser_args['conv_oversamp'])#this ensures a deep copy
    params.phase_centre_ra = base_types.uvw_ctypes_convert_type(data._field_centres[parser_args['field_id'],0,0]) #this ensures a deep copy
    params.phase_centre_dec = base_types.uvw_ctypes_convert_type(data._field_centres[parser_args['field_id'],0,1]) #this ensures a deep copy
    params.should_invert_jones_terms = ctypes.c_bool(parser_args['do_jones_corrections']) #this ensures a deep copy
    params.imaging_field = ctypes.c_uint(parser_args['field_id']) #this ensures a deep copy
    params.channel_grid_indicies = channel_grid_index.ctypes.data_as(ctypes.c_void_p) #this won't change between chunks
    params.cube_channel_dim_size = ctypes.c_size_t(cube_chan_dim_size) #this won't change between chunks
    params.output_buffer = gridded_vis.ctypes.data_as(ctypes.c_void_p) #we never do 2 computes at the same time (or the reduction is handled at the C++ implementation level)
    params.baseline_count = ctypes.c_size_t(data._no_baselines) #this ensures a deep copy
    params.number_of_polarization_terms = ctypes.c_size_t(data._no_polarization_correlations) #this ensures a deep copy
    params.number_of_polarization_terms_being_gridded = ctypes.c_size_t(len(correlations_to_grid))
    params.spw_count = ctypes.c_size_t(data._no_spw) #this ensures a deep copy
    params.channel_count = ctypes.c_size_t(data._no_channels) #this ensures a deep copy
    params.antenna_count = ctypes.c_size_t(data._no_antennae) #this ensures a deep copy
    params.enabled_channels = enabled_channels.ctypes.data_as(ctypes.c_void_p) #this won't change between chunks
    params.reference_wavelengths = data._chan_wavelengths.ctypes.data_as(ctypes.c_void_p) #this is part of the header of the MS and must stay constant between chunks
    params.should_grid_sampling_function = ctypes.c_bool(parser_args['output_psf'] or (parser_args['sample_weighting'] == 'uniform'))
    if parser_args['output_psf'] or (parser_args['sample_weighting'] == 'uniform'):
      params.sampling_function_buffer = sampling_funct.ctypes.data_as(ctypes.c_void_p) #we never do 2 computes at the same time (or the reduction is handled at the C++ implementation level)
      params.sampling_function_channel_grid_indicies = sampling_function_channel_grid_index.ctypes.data_as(ctypes.c_void_p) #this won't change between chunks
      params.sampling_function_channel_count = ctypes.c_size_t(sampling_function_channel_count) #this won't change between chunks

    params.num_facet_centres = ctypes.c_size_t(max(1,num_facet_centres)) #stays constant between strides
    params.facet_centres = facet_centres.ctypes.data_as(ctypes.c_void_p)

    if len(correlations_to_grid) == 1 and not parser_args['do_jones_corrections']:
	pol_index = data._polarization_correlations.tolist().index(correlations_to_grid[0])
	params.polarization_index = ctypes.c_size_t(pol_index) #stays constant between strides
    elif len(correlations_to_grid) == 2 and not parser_args['do_jones_corrections']: #the user want to derive one of the stokes terms (I,Q,U,V)
	pol_index = data._polarization_correlations.tolist().index(correlations_to_grid[0])
	pol_index_2 = data._polarization_correlations.tolist().index(correlations_to_grid[1])
	params.polarization_index = ctypes.c_size_t(pol_index) #stays constant between strides
	params.second_polarization_index = ctypes.c_size_t(pol_index_2) #stays constant between strides
    else:#the user want to apply corrective terms
	pass
    #pass in the necessary parameters for w-projection
    params.wplanes = ctypes.c_size_t(parser_args['wplanes'])
    params.wmax_est = base_types.uvw_ctypes_convert_type(w_max)
    libimaging.initLibrary(ctypes.byref(params))

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
      data.read_data(start_row=chunk_lbound,no_rows=chunk_linecount,data_column = parser_args['data_column'],
		       do_romein_baseline_ordering=(parser_args['use_back_end'] == 'GPU'))

      '''
      after the compute of the previous cycle finishes make deep copies and
      fill out the common parameters for gridding this cycle:
      '''
      libimaging.gridding_barrier()   #WARNING: This async barrier is cricical for valid gridding results

      arr_data_cpy = data._arr_data #gridding will operate on deep copied memory
      params.visibilities = arr_data_cpy.ctypes.data_as(ctypes.c_void_p)
      arr_uvw_cpy = data._arr_uvw #gridding will operate on deep copied memory
      params.uvw_coords = arr_uvw_cpy.ctypes.data_as(ctypes.c_void_p)
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
      params.row_count = ctypes.c_size_t(chunk_linecount)
      if parser_args['do_jones_corrections']: #do faceting with jones corrections
	params.no_timestamps_read = ctypes.c_size_t(data._no_timestamps_read)
      params.is_final_data_chunk = ctypes.c_bool(chunk_index == no_chunks - 1)
      arr_antenna_1_cpy = data._arr_antenna_1 #gridding will operate with deep copied data
      params.antenna_1_ids = arr_antenna_1_cpy.ctypes.data_as(ctypes.c_void_p)
      arr_antenna_2_cpy = data._arr_antenna_2 #gridding will operate with deep copied data
      params.antenna_2_ids = arr_antenna_2_cpy.ctypes.data_as(ctypes.c_void_p)
      arr_time_indicies_cpy = data._time_indicies #gridding will operate with deep copied data
      params.timestamp_ids = arr_time_indicies_cpy.ctypes.data_as(ctypes.c_void_p)
      if parser_args['use_back_end'] == 'GPU':
	with data_set_loader.data_set_loader.time_to_load_chunks:
	  starting_indexes = np.zeros([data._no_baselines+1],dtype=np.intp) #this must be n(n-1)/2+n+1 since we want to be able to compute the number of timestamps for the last baseline
	  params.baseline_starting_indexes = starting_indexes.ctypes.data_as(ctypes.c_void_p)
	  libimaging.repack_input_data(ctypes.byref(params))
      '''
      no need to grid more than one of the correlations if the user isn't interrested in imaging one of the stokes terms (I,Q,U,V) or the stokes terms are the correlation products:
      '''
      if len(correlations_to_grid) == 1 and not parser_args['do_jones_corrections']:
	if (num_facet_centres == 0):
	  libimaging.grid_single_pol(ctypes.byref(params))
	else: #facet single correlation term
	  libimaging.facet_single_pol(ctypes.byref(params))
      elif len(correlations_to_grid) == 2 and not parser_args['do_jones_corrections']: #the user want to derive one of the stokes terms (I,Q,U,V)
	if (num_facet_centres == 0):
	  libimaging.grid_duel_pol(ctypes.byref(params))
	else: #facet single correlation term
	  libimaging.facet_duel_pol(ctypes.byref(params))
      else: #the user want to apply corrective terms
	if (num_facet_centres == 0): #don't do faceting
	  libimaging.grid_4_cor(ctypes.byref(params))
	else:
	  if parser_args['do_jones_corrections']: #do faceting with jones corrections
	    jones_terms_cpy = data._jones_terms #gridding will operate with deep copied data
	    params.jones_terms = jones_terms_cpy.ctypes.data_as(ctypes.c_void_p)
	    libimaging.facet_4_cor_corrections(ctypes.byref(params))
	  else: #skip the jones corrections
	    libimaging.facet_4_cor(ctypes.byref(params))
      '''
      Now grid the psfs
      '''
      if parser_args['output_psf'] or (parser_args['sample_weighting'] == 'uniform'):
	if (num_facet_centres == 0):
	  libimaging.grid_sampling_function(ctypes.byref(params))
	else:
	  libimaging.facet_sampling_function(ctypes.byref(params))

  '''
  before compacting everything we better normalize
  '''
  if parser_args['sample_weighting'] == 'uniform':
    libimaging.weight_uniformly(ctypes.byref(params))
  libimaging.normalize(ctypes.byref(params))

  '''
  Compute the stokes term from the gridded visibilities (here we're passing a view)
  '''
  command_line_options.create_stokes_term_from_gridded_vis(parser_args['do_jones_corrections'],
                                                           data,
                                                           correlations_to_grid,
                                                           feeds_in_use,
                                                           gridded_vis.view(),
                                                           parser_args['pol'])

  '''
  now compact all the grids per facet into continuous blocks [channels,nx,ny]
  ie. remove the extra temporary correlation term grids per facet. There will
  still be some space left between the facets
  '''
  if parser_args['pol'] in ["I","Q","U","V"]:
    for f in range(0,max(1,num_facet_centres)):
      for c in range(0,cube_chan_dim_size):
	shift_count = (len(correlations_to_grid) - 1) * c # there are pol-1 blank spaces between channel grids
	nf = shift_count % len(correlations_to_grid)
	nc = shift_count / len(correlations_to_grid)
	gridded_vis[f,nc,nf,:,:] = gridded_vis[f,c,0,:,:]

  '''
  now finalize images
  '''
  libimaging.finalize(ctypes.byref(params))

  if parser_args['output_psf']:
    libimaging.finalize_psf(ctypes.byref(params))

  '''
  finally we can write to disk
  '''
  for f in range(0, max(1,num_facet_centres)):
    image_prefix = parser_args['output_prefix'] if num_facet_centres == 0 else parser_args['output_prefix']+"_facet"+str(f)
    if parser_args['output_format'] == 'png':
      offset = len(correlations_to_grid)*npix_l*npix_l*f*np.dtype(np.float32).itemsize
      dirty = np.ctypeslib.as_array(ctypes.cast(gridded_vis.ctypes.data + offset, ctypes.POINTER(ctypes.c_float)),
				    shape=(npix_l,npix_m))
      png_export.png_export(dirty[l_left_margin:l_right_margin,m_left_margin:m_right_margin],image_prefix,None)
      if parser_args['open_default_viewer']:
	os.system("xdg-open %s.png" % image_prefix)
      if parser_args['output_psf']:
	for i,c in enumerate(channels_to_image):
	  offset = npix_m*npix_l*f*np.dtype(np.float32).itemsize
	  psf = np.ctypeslib.as_array(ctypes.cast(sampling_funct.ctypes.data + offset, ctypes.POINTER(ctypes.c_float)),
				      shape=(npix_l,npix_m))
	  psf /= np.max(psf)
	  spw_no = c / data._no_channels
	  chan_no = c % data._no_channels
	  png_export.png_export(psf[l_left_margin:l_right_margin,m_left_margin:m_right_margin],
				image_prefix+('.spw%d.ch%d.psf' % (spw_no,chan_no)),None)

    else: #export to FITS cube
      ra = data._field_centres[parser_args['field_id'],0,0]
      dec = data._field_centres[parser_args['field_id'],0,1]
      offset_coord_l = (0 if num_facet_centres == 0 else facet_centres[f,0] - ra) / parser_args['cell_l']
      centre_coord_l = (parser_args['npix_l'] * 0.5 + 1) + offset_coord_l
      offset_coord_m = (0 if num_facet_centres == 0 else facet_centres[f,1] - dec) / parser_args['cell_m']
      centre_coord_m = (parser_args['npix_m'] * 0.5 + 1) - offset_coord_m
      offset = cube_chan_dim_size*len(correlations_to_grid)*npix_l*npix_m*f*np.dtype(np.float32).itemsize
      dirty = np.ctypeslib.as_array(ctypes.cast(gridded_vis.ctypes.data + offset, ctypes.POINTER(ctypes.c_float)),
				    shape=(cube_chan_dim_size,npix_l,npix_m))

      fits_export.save_to_fits_image(image_prefix+'.fits',
				     parser_args['npix_l'],parser_args['npix_m'],
				     quantity(parser_args['cell_l'],'arcsec'),quantity(parser_args['cell_m'],'arcsec'),
				     centre_coord_l,centre_coord_m,
				     quantity(ra,'arcsec'),
				     quantity(dec,'arcsec'),
				     parser_args['pol'],
				     cube_first_wavelength,
				     cube_delta_wavelength,
				     cube_chan_dim_size,
				     dirty[:,l_left_margin:l_right_margin,m_left_margin:m_right_margin])
      if parser_args['open_default_viewer']:
	os.system("xdg-open %s.fits" % image_prefix)
      if parser_args['output_psf']:
	for i,c in enumerate(channels_to_image):
	  offset = i*npix_l*npix_m*f*np.dtype(np.float32).itemsize
	  psf = np.ctypeslib.as_array(ctypes.cast(sampling_funct.ctypes.data + offset, ctypes.POINTER(ctypes.c_float)),
				      shape=(1,npix_l,npix_m))
	  psf /= np.max(psf)
	  spw_no = c / data._no_channels
	  chan_no = c % data._no_channels
	  ra = data._field_centres[parser_args['field_id'],0,0]
	  dec = data._field_centres[parser_args['field_id'],0,1]
	  offset_coord_l = (0 if num_facet_centres == 0 else facet_centres[f,0] - ra) / parser_args['cell_l']
	  centre_coord_l = (parser_args['npix_l'] * 0.5 + 1) + offset_coord_l
	  offset_coord_m = (0 if num_facet_centres == 0 else facet_centres[f,1] - dec) / parser_args['cell_m']
	  centre_coord_m = (parser_args['npix_m'] * 0.5 + 1) - offset_coord_m
	  fits_export.save_to_fits_image(image_prefix+('.spw%d.ch%d.psf.fits' % (spw_no,chan_no)),
					 parser_args['npix_l'],parser_args['npix_m'],
					 quantity(parser_args['cell_l'],'arcsec'),quantity(parser_args['cell_m'],'arcsec'),
					 centre_coord_l,centre_coord_m,
					 quantity(ra,'arcsec'),
					 quantity(dec,'arcsec'),
					 parser_args['pol'],
					 data._chan_wavelengths[spw_no,chan_no],
					 0,
					 1,
					 psf[:,l_left_margin:l_right_margin,m_left_margin:m_right_margin])

  '''
  attempt to stitch the facets together:
  '''
  if parser_args['stitch_facets']: #we've already checked that there are multiple facets before this line
    print "ATTEMPTING TO MOSAIC FACETS USING MONTAGE (MUST BE IN ENV PATH)"
    facet_list_parser.output_mosaic(parser_args['output_prefix'],num_facet_centres)
    if parser_args['open_default_viewer']:
	os.system("xdg-open %s.combined.fits" % parser_args['output_prefix'])

  print "FINISHED WORK SUCCESSFULLY"
  total_run_time.stop()
  print "STATISTICS:"
  print "\tConvolution filter and detapering function creation time: %f secs" % filter_creation_timer.elapsed()
  print "\tIn Parallel:"
  print "\t\tData loading and conversion time: %f secs" % data_set_loader.data_set_loader.time_to_load_chunks.elapsed()
  libimaging.get_gridding_walltime.restype = ctypes.c_double
  libimaging.get_inversion_walltime.restype = ctypes.c_double
  print "\t\tGridding time: %f secs" % libimaging.get_gridding_walltime()
  print "\tFourier inversion time: %f secs" % libimaging.get_inversion_walltime()
  print "\tTotal runtime: %f secs" % total_run_time.elapsed()
  libimaging.releaseLibrary()
  exit(0)