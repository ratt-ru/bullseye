#!/usr/bin/python
import sys
import os
import shutil
import argparse
from pyparsing import commaSeparatedList
import numpy as np
import pylab
import re
from os.path import *
from pyrap.quanta import quantity

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
#libimaging = ctypes.pydll.LoadLibrary("build/gpu_algorithm/libgpu_imaging.so")
def coords(s):  
    try:
	sT = s.strip()
        ra, dec = map(float, sT[1:len(sT)-1].split(','))
        return ra, dec
    except:
        raise argparse.ArgumentTypeError("Coordinates must be ra,dec tupples")
      
def channel_range(s):
    sT = s.strip()
    if re.match('[0-9]+:[0-9]+(~[0-9]+)?(,[0-9]+(~[0-9]+)?)*$',sT) != None:
	spw,selections = s.split(':')
	ranges = selections.split(',')
	channels = []
	for r in ranges:
	  if re.match('[0-9]+~[0-9]+$',r) != None:
	    start,finish = r.split('~')
	    channels = channels + range(int(start),int(finish)+1)
	  else:
	    channels.append(int(r))
	return int(spw),sorted(set(channels))
    else:
	raise argparse.ArgumentTypeError("Channel ranges should be specified as 'spw index':'comma seperated ranges of channels', for example 0:0,2~5,7 will select channels 0,2,3,4,5,7 from spw 0")
      
if __name__ == "__main__":
  total_run_time = timer.timer()
  total_run_time.start()
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
  parser.add_argument('--facet_centres', help='List of coordinate tupples indicating facet centres (RA,DEC). ' 
		      'If none are specified and n_facet_l and/or n_facet_m are not set, the default pointing centre will be used', type=coords, nargs='+', default=None)
  parser.add_argument('--npix_l', help='Number of facet pixels in l', type=int, default=256)
  parser.add_argument('--npix_m', help='Number of facet pixels in m', type=int, default=256)
  parser.add_argument('--cell_l', help='Size of a pixel in l (arcsecond)', type=float, default=1)
  parser.add_argument('--cell_m', help='Size of a pixel in m (arcsecond)', type=float, default=1)
  parser.add_argument('--pol', help='Specify image polarization', choices=pol_options.keys(), default="XX")
  parser.add_argument('--conv', help='Specify gridding convolution function type', choices=['gausian','keiser bessel'], default='keiser bessel')
  parser.add_argument('--conv_sup', help='Specify gridding convolution function support area (number of grid cells)', type=int, default=1)
  parser.add_argument('--conv_oversamp', help='Specify gridding convolution function oversampling multiplier', type=int, default=7)
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
					       "for example --channel_select (0:1,3~5,7) (1:2) will select channels 1,3,4,5,7 from spw 0 and channel 2 from spw 1. Default all",
		      type=channel_range, nargs='+', default=None)
  parser.add_argument('--average_spw_channels', help='Averages selected channels in each spectral window', type=bool, default=False)
  parser.add_argument('--average_all', help='Averages all selected channels together into a single image', type=bool, default=False)
  parser.add_argument('--output_psf',help='Outputs the Point Spread Function (per channel)',type=bool,default=False)
  parser.add_argument('--sample_weighting',help='Specify weighting technique in use.',choices=['natural','uniform'], default='natural')
  
  parser_args = vars(parser.parse_args())
  '''
  initially the output grids must be set to NONE. Memory will only be allocated before the first MS is read.
  '''
  gridded_vis = None
  sampling_funct = None
  ms_names = commaSeparatedList.parseString(parser_args['input_ms'])
  for ms_index,ms in enumerate(ms_names):
    print "NOW IMAGING %s" % ms
    data = data_set_loader.data_set_loader(ms,read_jones_terms=parser_args['do_jones_corrections'])
    data.read_head()
    no_chunks = parser_args['no_chunks']
    if no_chunks < 1:
      raise Exception("Cannot read less than one chunk from measurement set!")
    chunk_size = int(np.ceil(data._no_rows / float(no_chunks)))
    
    '''
    some sanity checks:
    check that the measurement set contains the correlation terms necessary to create the requested pollarization / Stokes term:
    '''
    correlations_to_grid = None
    feeds_in_use = None
    if parser_args['do_jones_corrections']:
      correlations_to_grid = data._polarization_correlations.tolist() #check below if this list is supported
    else:
      for feed_index,req in enumerate(pol_dependencies[parser_args['pol']]):
	if set(req).issubset(data._polarization_correlations.tolist()):
	  correlations_to_grid = req #we found a subset that must be gridded
	  feeds_in_use = feed_descriptions[parser_args['pol']][feed_index]
    if correlations_to_grid == None:
      raise argparse.ArgumentTypeError("Cannot obtain requested gridded polarization from the provided measurement set. Need one of %s" % pol_dependencies[parser_args['pol']])
    '''
    check that we have 4 correlations before trying to apply jones corrective terms:
    '''
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
    '''
    check how many facets we have to create (if none we'll do normal gridding without any transformations)
    '''
    num_facet_centres = parser_args['n_facets_l'] * parser_args['n_facets_m']
    if (parser_args['facet_centres'] != None):
      num_facet_centres += len(parser_args['facet_centres'])
    if parser_args['stitch_facets']:
      if num_facet_centres < 2:
	raise argparse.ArgumentTypeError("Need at least two facets to perform stitching")
      if parser_args['output_format'] != 'fits':
	raise argparse.ArgumentTypeError("Facet output format must be of type FITS")
    facet_centres = np.empty([parser_args['n_facets_m'],parser_args['n_facets_l'],2],dtype=base_types.uvw_type)
    facet_centres[:parser_args['n_facets_m'],:parser_args['n_facets_l'],0] = np.tile((np.arange(0,parser_args['n_facets_l']) + 1 - np.ceil(parser_args['n_facets_l']/2.0))*parser_args['npix_l']*parser_args['cell_l'] +
										     data._field_centres[parser_args['field_id'],0,0],[parser_args['n_facets_m'],1])
    facet_centres[:parser_args['n_facets_m'],:parser_args['n_facets_l'],1] = np.repeat((np.arange(0,parser_args['n_facets_m']) + 1 - np.ceil(parser_args['n_facets_m']/2.0))*parser_args['npix_m']*parser_args['cell_m'] +
										     data._field_centres[parser_args['field_id'],0,1],parser_args['n_facets_l']).reshape(parser_args['n_facets_m'],parser_args['n_facets_l'])
    facet_centres = facet_centres.reshape(parser_args['n_facets_l']*parser_args['n_facets_m'],2)
    
    if (parser_args['facet_centres'] != None):
      facet_centres = np.append(facet_centres,np.array(parser_args['facet_centres']).astype(base_types.uvw_type)).reshape(num_facet_centres,2)
    if num_facet_centres != 0:
      print "REQUESTED FACET CENTRES:"
      for i,c in enumerate(facet_centres):
	print "\tFACET %d RA: %s DEC: %s" % (i,quantity(c[0],'arcsec').get('deg'),quantity(c[1],'arcsec').get('deg'))
    
    if parser_args['do_jones_corrections'] and num_facet_centres != data._cal_no_dirs:
      raise argparse.ArgumentTypeError("Number of calibrated directions does not correspond to number of directions being faceted")
    '''
    populate the channels to be imaged:
    '''
    channels_to_image = []
    if parser_args['channel_select'] == None:
      channels_to_image = channels_to_image + range(0,data._no_spw*data._no_channels)
    else:
      for spw,channels in parser_args['channel_select']:
	channels_to_image = channels_to_image + map(lambda x:x+spw*data._no_channels,channels)
    channels_to_image = sorted(set(channels_to_image)) #remove duplicates and sort
    enabled_channels = np.array([False for i in range(0,data._no_spw*data._no_channels)])
    for c in channels_to_image:
      spw_no = c / data._no_channels
      chan_no = c % data._no_channels
      enabled_channels[spw_no*data._no_channels + chan_no] = True
    
    if channels_to_image[len(channels_to_image)-1] >= data._no_spw*data._no_channels:
      raise argparse.ArgumentTypeError("One or more channels don't exist. Only %d spw's of %d channels each are available" % (data._no_spw,data._no_channels))
    print "REQUESTED THE FOLLOWING CHANNELS BE IMAGED:"
    for c in channels_to_image:
      spw_no = c / data._no_channels
      chan_no = c % data._no_channels
      print "\tSPW %d CHANNEL %d AT WAVELENGTH %f" % (spw_no,chan_no,data._chan_wavelengths[spw_no,chan_no])
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
      spw_0_no = channels_to_image[0] / data._no_channels
      chan_0_no = channels_to_image[0] % data._no_channels
      spw_1_no = channels_to_image[1] / data._no_channels
      chan_1_no = channels_to_image[1] % data._no_channels
      cube_delta_wavelength = data._chan_wavelengths[spw_1_no,chan_1_no] - data._chan_wavelengths[spw_0_no,chan_0_no]
      cube_first_wavelength = data._chan_wavelengths[spw_0_no,chan_0_no]
      epsilon = 0.00000000000001
      for i in range(2,len(channels_to_image)): #first delta calculated, now loop through the remainder of the enabled channels and check if the deltas match
	  spw_0_no = channels_to_image[i-1] / data._no_channels
	  chan_0_no = channels_to_image[i-1] % data._no_channels
	  spw_1_no = channels_to_image[i] / data._no_channels
	  chan_1_no = channels_to_image[i] % data._no_channels
	  if (abs(data._chan_wavelengths[spw_1_no,chan_1_no] - 
		 data._chan_wavelengths[spw_0_no,chan_0_no] - 
		 cube_delta_wavelength) > epsilon):
	    raise argparse.ArgumentTypeError("Selected channels are not evenly spaced. Cannot create a fits cube from them. "
					     "Try averaging per spectral window (--average_spw_channels) or all (--average_all).")
    elif len(channels_to_image) > 1 and parser_args['average_spw_channels']:
      first_spw=True
      for i in range(0,len(channels_to_image)-1):
	spw_0 = channels_to_image[i] / data._no_channels
	spw_1 = channels_to_image[i+1] / data._no_channels
	if spw_1 > spw_0: #loop until we come accross the border between SPWs (remember each spw can have 0 <= x <= no_channels enabled)
	  if first_spw:
	    cube_delta_wavelength = data._spw_centres[spw_1] - data._spw_centres[spw_0]
	    cube_first_wavelength = data._spw_centres[spw_0]
	    first_spw = False
	    
	  epsilon = 0.00000000000001
	  if abs(data._spw_centres[spw_1] - data._spw_centres[spw_0] - cube_delta_wavelength) > epsilon:
	    raise argparse.ArgumentTypeError("Consecutive spectral windows are not evenly spaced. Cannot create a fits cube from them. "
					     "Try imaging one spectral window at a time or average all spws (--average_all).")
    elif parser_args['average_all']:
      cube_delta_wavelength = 0
      cube_first_wavelength = data._chan_wavelengths.mean()
    else: #only one channel
      spw_0_no = channels_to_image[0] / data._no_channels
      chan_0_no = channels_to_image[0] % data._no_channels
      cube_delta_wavelength = 0
      cube_first_wavelength = data._chan_wavelengths[spw_0_no,chan_0_no]
      
    channel_grid_index = np.zeros([data._no_spw*data._no_channels],dtype=np.intp) #stores the index of the grid this channel should be saved to (usage: image cubes)
    cube_chan_dim_size = 0
    '''
    work out how many channels / averaged SPWs this cube will have (remember each SPW may have 0 <= x <= no_channels enabled)
    '''
    if parser_args['average_spw_channels']:
      cube_chan_dim_size += 1 #at least one channel, so at least one spw
      current_spw = channels_to_image[0] / data._no_channels
      current_grid = 0
      for c in range(0,len(enabled_channels)):
	channel_grid_index[c] = current_grid
	if enabled_channels[c] and (c / data._no_channels) > current_spw:
	  cube_chan_dim_size += 1
	  current_grid += 1
	  current_spw = c / data._no_channels
    elif len(channels_to_image) > 1 and not parser_args['average_all']: #grid individual channels
      current_grid = 0
      for c in range(0,len(enabled_channels)):
	channel_grid_index[c] = current_grid
	if enabled_channels[c]:
	  current_grid += 1
	  cube_chan_dim_size += 1
	
    else:
      cube_chan_dim_size = 1
    '''
    Work out to which grid each sampling function (per channel) should be gridded.
    '''
    sampling_function_channel_grid_index = np.zeros([data._no_spw*data._no_channels],dtype=np.intp)
    sampling_function_channel_count = len(channels_to_image)
    if parser_args['output_psf'] or parser_args['sample_weighting'] != 'natural':
      current_grid = 0
      for c in range(0,len(enabled_channels)):
	sampling_function_channel_grid_index[c] = current_grid
	if enabled_channels[c]:
	  current_grid += 1
   
    '''
    allocate enough memory to compute image and or facets (only before gridding the first MS)
    '''
    if not parser_args['do_jones_corrections']:
	if gridded_vis == None:
	  num_facet_grids = 1 if (num_facet_centres == 0) else num_facet_centres
	  gridded_vis = np.zeros([num_facet_grids,cube_chan_dim_size,len(correlations_to_grid),parser_args['npix_l'],parser_args['npix_m']],dtype=base_types.grid_type)
    else:
	if gridded_vis == None:
	  num_facet_grids = 1 if (num_facet_centres == 0) else num_facet_centres
	  gridded_vis = np.zeros([num_facet_grids,cube_chan_dim_size,4,parser_args['npix_l'],parser_args['npix_m']],dtype=base_types.grid_type)
    
    if sampling_funct == None:
	  if parser_args['output_psf'] or parser_args['sample_weighting'] != 'natural':
	    num_facet_grids = 1 if (num_facet_centres == 0) else num_facet_centres
	    sampling_funct = np.zeros([num_facet_grids,sampling_function_channel_count,1,parser_args['npix_l'],parser_args['npix_m']],dtype=base_types.psf_type)
    '''
    initiate the backend imaging library
    '''
    params = gridding_parameters.gridding_parameters()
    params.chunk_max_row_count = ctypes.c_size_t(chunk_size)
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
    if parser_args['output_psf'] or parser_args['sample_weighting'] != 'natural':
      params.polarization_index = ctypes.c_size_t(0) #stays constant between strides
      params.sampling_function_buffer = sampling_funct.ctypes.data_as(ctypes.c_void_p) #we never do 2 computes at the same time (or the reduction is handled at the C++ implementation level)
      params.sampling_function_channel_grid_indicies = sampling_function_channel_grid_index.ctypes.data_as(ctypes.c_void_p) #this won't change between chunks
      params.sampling_function_channel_count = ctypes.c_size_t(sampling_function_channel_count) #this won't change between chunks
    params.num_facet_centres = ctypes.c_size_t(max(1,num_facet_centres)) #stays constant between strides
    params.facet_centres = facet_centres.ctypes.data_as(ctypes.c_void_p)
    params.detapering_buffer = conv._F_detaper.ctypes.data_as(ctypes.c_void_p) #stays constant between strides
    
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
      data.read_data(start_row=chunk_lbound,no_rows=chunk_linecount,data_column = parser_args['data_column'])
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
      params.no_timestamps_read = ctypes.c_size_t(data._no_timestamps_read)
      params.is_final_data_chunk = ctypes.c_bool(chunk_index == no_chunks - 1)
      arr_antenna_1_cpy = data._arr_antenna_1 #gridding will operate with deep copied data
      params.antenna_1_ids = arr_antenna_1_cpy.ctypes.data_as(ctypes.c_void_p)
      arr_antenna_2_cpy = data._arr_antenna_2 #gridding will operate with deep copied data
      params.antenna_2_ids = arr_antenna_2_cpy.ctypes.data_as(ctypes.c_void_p)
      arr_time_indicies_cpy = data._time_indicies #gridding will operate with deep copied data
      params.timestamp_ids = arr_time_indicies_cpy.ctypes.data_as(ctypes.c_void_p)
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
      Now grid the psfs (this will automatically wait till visibility gridding has been completed)
      '''
      if parser_args['output_psf'] or parser_args['sample_weighting'] != 'natural':
	if (num_facet_centres == 0):
	  libimaging.grid_sampling_function(ctypes.byref(params))
	else:
	  libimaging.facet_sampling_function(ctypes.byref(params))
	  
      if chunk_index == no_chunks - 1:
	libimaging.gridding_barrier()
	if parser_args['sample_weighting'] == 'uniform':
	  libimaging.weight_uniformly(ctypes.byref(params))
	
  '''
  See Smirnov I (2011) for description on conversion between correlation terms and stokes params for linearly polarized feeds
  See Synthesis Imaging II (1999) Pg. 9 for a description on conversion between correlation terms and stokes params for circularly polarized feeds
  '''
  if parser_args['do_jones_corrections']:
    if feeds_in_use == 'circular':		#circular correlation products
      if parser_args['pol'] == "I":
	IpV = data._polarization_correlations.tolist().index(pol_options['RR'])
	ImV = data._polarization_correlations.tolist().index(pol_options['LL'])
	gridded_vis[:,:,0,:,:] = ((gridded_vis[:,:,IpV,:,:] + gridded_vis[:,:,ImV,:,:])/2)
      elif parser_args['pol'] == "V":
	IpV = data._polarization_correlations.tolist().index(pol_options['RR'])
	ImV = data._polarization_correlations.tolist().index(pol_options['LL'])
	gridded_vis[:,:,0,:,:] = ((gridded_vis[:,:,IpV,:,:] - gridded_vis[:,:,ImV,:,:])/2)
      elif parser_args['pol'] == "Q":
	QpiU = data._polarization_correlations.tolist().index(pol_options['RL'])
	QmiU = data._polarization_correlations.tolist().index(pol_options['LR'])
	gridded_vis[:,:,0,:,:] = ((gridded_vis[:,:,QpiU,:,:] + gridded_vis[:,:,QmiU,:,:])/2)
      elif parser_args['pol'] == "U":
	QpiU = data._polarization_correlations.tolist().index(pol_options['RL'])
	QmiU = data._polarization_correlations.tolist().index(pol_options['LR'])
	gridded_vis[:,:,0,:,:] = ((gridded_vis[:,:,QpiU,:,:] - gridded_vis[:,:,QmiU,:,:])/2.0)
      elif parser_args['pol'] in ['RR','RL','LR','LL']:
	pass #already stored in gridded_vis[:,0,:,:]
    elif feeds_in_use == 'linear':		#linear correlation products
      if parser_args['pol'] == "I":
	IpQ = data._polarization_correlations.tolist().index(pol_options['XX'])
	ImQ = data._polarization_correlations.tolist().index(pol_options['YY'])
	gridded_vis[:,:,0,:,:] = ((gridded_vis[:,:,IpQ,:,:] + gridded_vis[:,:,ImQ,:,:]))
      elif parser_args['pol'] == "Q":
	IpQ = data._polarization_correlations.tolist().index(pol_options['XX'])
	ImQ = data._polarization_correlations.tolist().index(pol_options['YY'])
	gridded_vis[:,:,0,:,:] = ((gridded_vis[:,:,IpQ,:,:] - gridded_vis[:,:,ImQ,:,:]))
      elif parser_args['pol'] == "U":
	UpiV = data._polarization_correlations.tolist().index(pol_options['XY'])
	UmiV = data._polarization_correlations.tolist().index(pol_options['YX'])
	gridded_vis[:,:,0,:,:] = ((gridded_vis[:,:,UpiV,:,:] + gridded_vis[:,:,UmiV,:,:]))
      elif parser_args['pol'] == "V":
	UpiV = data._polarization_correlations.tolist().index(pol_options['XY'])
	UmiV = data._polarization_correlations.tolist().index(pol_options['YX'])
	gridded_vis[:,:,0,:,:] = ((gridded_vis[:,:,UpiV,:,:] - gridded_vis[:,:,UmiV,:,:]))/1.0
      elif parser_args['pol'] in ['XX','XY','YX','YY']:
	pass #already stored in gridded_vis[:,0,:,:]
    else:
      pass #any cases not stated here should be flagged by sanity checks on the program arguement list
  else:
    if feeds_in_use == 'circular':		#circular correlation products
      if parser_args['pol'] == "I":
	IpV = correlations_to_grid.index(pol_options['RR'])
	ImV = correlations_to_grid.index(pol_options['LL'])
	gridded_vis[:,:,0,:,:] = ((gridded_vis[:,:,IpV,:,:] + gridded_vis[:,:,ImV,:,:])/2)
      elif parser_args['pol'] == "V":
	IpV = correlations_to_grid.index(pol_options['RR'])
	ImV = correlations_to_grid.index(pol_options['LL'])
	gridded_vis[:,:,0,:,:] = ((gridded_vis[:,:,IpV,:,:] - gridded_vis[:,:,ImV,:,:])/2)
      elif parser_args['pol'] == "Q":
	QpiU = correlations_to_grid.index(pol_options['RL'])
	QmiU = correlations_to_grid.index(pol_options['LR'])
	gridded_vis[:,:,0,:,:] = ((gridded_vis[:,:,QpiU,:,:] + gridded_vis[:,:,QmiU,:,:])/2)
      elif parser_args['pol'] == "U":
	QpiU = correlations_to_grid.index(pol_options['RL'])
	QmiU = correlations_to_grid.index(pol_options['LR'])
	gridded_vis[:,:,0,:,:] = ((gridded_vis[:,:,QpiU,:,:] - gridded_vis[:,:,QmiU,:,:])/2)
      elif parser_args['pol'] in ['RR','RL','LR','LL']:
	pass #already stored in gridded_vis[:,0,:,:]
    elif feeds_in_use == 'linear':		#linear correlation products
      if parser_args['pol'] == "I":
	IpQ = correlations_to_grid.index(pol_options['XX'])
	ImQ = correlations_to_grid.index(pol_options['YY'])
	gridded_vis[:,:,0,:,:] = ((gridded_vis[:,:,IpQ,:,:] + gridded_vis[:,:,ImQ,:,:]))
      elif parser_args['pol'] == "Q":
	IpQ = correlations_to_grid.index(pol_options['XX'])
	ImQ = correlations_to_grid.index(pol_options['YY'])
	gridded_vis[:,:,0,:,:] = ((gridded_vis[:,:,IpQ,:,:] - gridded_vis[:,:,ImQ,:,:]))
      elif parser_args['pol'] == "U":
	UpiV = correlations_to_grid.index(pol_options['XY'])
	UmiV = correlations_to_grid.index(pol_options['YX'])
	gridded_vis[:,:,0,:,:] = ((gridded_vis[:,:,UpiV,:,:] + gridded_vis[:,:,UmiV,:,:]))
      elif parser_args['pol'] == "V":
	UpiV = correlations_to_grid.index(pol_options['XY'])
	UmiV = correlations_to_grid.index(pol_options['YX'])
	gridded_vis[:,:,0,:,:] = ((gridded_vis[:,:,UpiV,:,:] - gridded_vis[:,:,UmiV,:,:]))
      elif parser_args['pol'] in ['XX','XY','YX','YY']:
	pass #already stored in gridded_vis[:,0,:,:]
    else:
      pass #any cases not stated here should be flagged by sanity checks on the program arguement list
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
  now invert, detaper and write out all the facets to disk:  
  '''
  libimaging.finalize(ctypes.byref(params))
  if parser_args['output_psf']:
    libimaging.finalize_psf(ctypes.byref(params))
  for f in range(0, max(1,num_facet_centres)):
    image_prefix = parser_args['output_prefix'] if num_facet_centres == 0 else parser_args['output_prefix']+"_facet"+str(f)
    if parser_args['output_format'] == 'png':
      offset = len(correlations_to_grid)*parser_args['npix_l']*parser_args['npix_m']*f*np.dtype(np.float32).itemsize
      dirty = np.ctypeslib.as_array(ctypes.cast(gridded_vis.ctypes.data + offset, ctypes.POINTER(ctypes.c_float)),
				    shape=(parser_args['npix_l'],parser_args['npix_m']))
      png_export.png_export(dirty,image_prefix,None)
      if parser_args['output_psf']:
	for i,c in enumerate(channels_to_image):
	  offset = parser_args['npix_l']*parser_args['npix_m']*f*np.dtype(np.float32).itemsize
	  psf = np.ctypeslib.as_array(ctypes.cast(sampling_funct.ctypes.data + offset, ctypes.POINTER(ctypes.c_float)),
				      shape=(parser_args['npix_l'],parser_args['npix_m']))
	  spw_no = c / data._no_channels
	  chan_no = c % data._no_channels
	  png_export.png_export(psf,image_prefix+('.spw%d.ch%d.psf' % (spw_no,chan_no)),None)
      
    else: #export to FITS cube
      ra = data._field_centres[parser_args['field_id'],0,0] if num_facet_centres == 0 else facet_centres[f,0]
      dec = data._field_centres[parser_args['field_id'],0,1] if num_facet_centres == 0 else facet_centres[f,1]
      offset = cube_chan_dim_size*len(correlations_to_grid)*parser_args['npix_l']*parser_args['npix_m']*f*np.dtype(np.float32).itemsize
      dirty = np.ctypeslib.as_array(ctypes.cast(gridded_vis.ctypes.data + offset, ctypes.POINTER(ctypes.c_float)),
				    shape=(cube_chan_dim_size,parser_args['npix_l'],parser_args['npix_m']))
      dirty /= parser_args['npix_l']*parser_args['npix_m'] #TODO FIX THIS
      fits_export.save_to_fits_image(image_prefix+'.fits',
				     parser_args['npix_l'],parser_args['npix_m'],
				     quantity(parser_args['cell_l'],'arcsec'),quantity(parser_args['cell_m'],'arcsec'),
				     quantity(ra,'arcsec'),
				     quantity(dec,'arcsec'),
				     parser_args['pol'],
				     cube_first_wavelength,
				     cube_delta_wavelength,
				     cube_chan_dim_size,
				     dirty)
      if parser_args['output_psf']:
	for i,c in enumerate(channels_to_image):
	  offset = i*parser_args['npix_l']*parser_args['npix_m']*f*np.dtype(np.float32).itemsize
	  psf = np.ctypeslib.as_array(ctypes.cast(sampling_funct.ctypes.data + offset, ctypes.POINTER(ctypes.c_float)),
				      shape=(1,parser_args['npix_l'],parser_args['npix_m']))
	  spw_no = c / data._no_channels
	  chan_no = c % data._no_channels
	  ra = data._field_centres[parser_args['field_id'],0,0] if num_facet_centres == 0 else facet_centres[f,0]
	  dec = data._field_centres[parser_args['field_id'],0,1] if num_facet_centres == 0 else facet_centres[f,1]
	  fits_export.save_to_fits_image(image_prefix+('.spw%d.ch%d.psf.fits' % (spw_no,chan_no)),
					 parser_args['npix_l'],parser_args['npix_m'],
					 quantity(parser_args['cell_l'],'arcsec'),quantity(parser_args['cell_m'],'arcsec'),
					 quantity(ra,'arcsec'),
					 quantity(dec,'arcsec'),
					 parser_args['pol'],
					 data._chan_wavelengths[spw_no,chan_no],
					 0,
					 1,
					 psf)
      
  '''
  attempt to stitch the facets together:
  '''
  if parser_args['stitch_facets']: #we've already checked that there are multiple facets before this line
    file_names = [basename(parser_args['output_prefix'] + '_facet' + str(i) + '.fits') for i in range(0,num_facet_centres)]
    facet_image_list_filename = dirname(parser_args['output_prefix']) + '/facets.lst'
    f_file_list = open(facet_image_list_filename,'w')
    f_file_list.writelines(["|%sfname|\n" % (" "*(max([len(item) for item in file_names])-5)),
			    "|%schar|\n" % (" "*(max([len(item) for item in file_names])-4))
			   ])
    f_file_list.writelines([" %s\n" % item for item in file_names])
    f_file_list.close()
    montage_unprojected_img_table = dirname(parser_args['output_prefix']) + '/facets.montage.tbl'
    os.system('mImgtbl -t %s %s %s' % (facet_image_list_filename,
				       dirname(parser_args['output_prefix']),
				       montage_unprojected_img_table
				      )
	     )
    montage_proj_template_hdr = dirname(parser_args['output_prefix']) + '/projected_template.hdr'
    os.system('mMakeHdr %s %s' % (montage_unprojected_img_table,
				  montage_proj_template_hdr
				 )
	     )
    proj_dir = dirname(parser_args['output_prefix']) + '/projected_facets'
    if exists(proj_dir):
      shutil.rmtree(proj_dir)
    os.makedirs(proj_dir)
    montage_stats_file = dirname(parser_args['output_prefix']) + '/stats.tbl'
    os.system('mProjExec -p %s %s %s %s %s' % (dirname(parser_args['output_prefix']),
					       montage_unprojected_img_table,
					       montage_proj_template_hdr,
					       proj_dir,
					       montage_stats_file
					      )
	     )
    montage_unprojected_img_table = dirname(parser_args['output_prefix']) + '/facets.montage.proj.tbl'
    os.system('mImgtbl %s %s' % (proj_dir,
				 montage_unprojected_img_table
				)
	     )
    montage_combined_img = parser_args['output_prefix'] + '.combined.fits'
    os.system('mAdd -p %s %s %s %s' % (proj_dir,
				       montage_unprojected_img_table,
				       montage_proj_template_hdr,
				       montage_combined_img
				      )
	     )
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
