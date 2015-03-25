import numpy as np
import argparse

def is_close(a,b,tolerance=1.0e-5):
  return abs(a - b) < tolerance

def compute_cube_chan_dim_spacing_no_averaging(data,channels_to_image):
  cube_first_wavelength = 0
  cube_delta_wavelength = 0
  spw_0_no = channels_to_image[0] / data._no_channels
  chan_0_no = channels_to_image[0] % data._no_channels
  spw_1_no = channels_to_image[1] / data._no_channels
  chan_1_no = channels_to_image[1] % data._no_channels
  cube_delta_wavelength = data._chan_wavelengths[spw_1_no,chan_1_no] - data._chan_wavelengths[spw_0_no,chan_0_no]
  cube_first_wavelength = data._chan_wavelengths[spw_0_no,chan_0_no]
  for i in range(2,len(channels_to_image)): #first delta calculated, now loop through the remainder of the enabled channels and check if the deltas match
    spw_0_no = channels_to_image[i-1] / data._no_channels
    chan_0_no = channels_to_image[i-1] % data._no_channels
    spw_1_no = channels_to_image[i] / data._no_channels
    chan_1_no = channels_to_image[i] % data._no_channels
    if not is_close(data._chan_wavelengths[spw_1_no,chan_1_no] - data._chan_wavelengths[spw_0_no,chan_0_no], 
		    cube_delta_wavelength):
      raise argparse.ArgumentTypeError("Selected channels are not evenly spaced. Cannot create a fits cube from them. "
				       "Try averaging per spectral window (--average_spw_channels) or all (--average_all).")
  return (cube_delta_wavelength,cube_first_wavelength)

def compute_cube_chan_dim_spw_averaging(data,channels_to_image):
  first_spw=True
  cube_first_wavelength = 0
  cube_delta_wavelength = 0
  for i in range(0,len(channels_to_image)-1):
    spw_0 = channels_to_image[i] / data._no_channels
    spw_1 = channels_to_image[i+1] / data._no_channels
    if spw_1 > spw_0: #loop until we come accross the border between SPWs (remember each spw can have 0 <= x <= no_channels enabled)
      if first_spw:
	cube_delta_wavelength = data._spw_centres[spw_1] - data._spw_centres[spw_0]
	cube_first_wavelength = data._spw_centres[spw_0]
	first_spw = False
	    
      if not is_close(data._spw_centres[spw_1] - data._spw_centres[spw_0],
		      cube_delta_wavelength):
	raise argparse.ArgumentTypeError("Consecutive spectral windows are not evenly spaced. Cannot create a fits cube from them. "
					 "Try imaging one spectral window at a time or average all spws (--average_all).")
  return (cube_delta_wavelength,cube_first_wavelength)

def compute_cube_chan_dim_all_channel_averaging(data):
  cube_delta_wavelength = 0
  cube_first_wavelength = data._chan_wavelengths.mean()
  return (cube_delta_wavelength,cube_first_wavelength)

def compute_cube_chan_dim_single_channel(data,channels_to_image):
  spw_0_no = channels_to_image[0] / data._no_channels
  chan_0_no = channels_to_image[0] % data._no_channels
  cube_delta_wavelength = 0
  cube_first_wavelength = data._chan_wavelengths[spw_0_no,chan_0_no]
  return (cube_delta_wavelength,cube_first_wavelength)

def parse_channels_to_be_imaged(user_channel_select,data):
  channels_to_image = []
  enabled_channels = np.array([False for i in range(0,data._no_spw*data._no_channels)])
  if user_channel_select == None:
    channels_to_image = channels_to_image + range(0,data._no_spw*data._no_channels)
  else:
    for spw,channels in user_channel_select:
      channels_to_image = channels_to_image + map(lambda x:x+spw*data._no_channels,channels)
  channels_to_image = sorted(set(channels_to_image)) #remove duplicates and sort
    
  for c in channels_to_image:
    spw_no = c / data._no_channels
    chan_no = c % data._no_channels
    enabled_channels[spw_no*data._no_channels + chan_no] = True
  return (channels_to_image,enabled_channels)

def print_requested_channels(channels_to_image,data):
  print "REQUESTED THE FOLLOWING CHANNELS BE IMAGED:"
  for c in channels_to_image:
    spw_no = c / data._no_channels
    chan_no = c % data._no_channels
    print "\tSPW %d CHANNEL %d AT WAVELENGTH %f" % (spw_no,chan_no,data._chan_wavelengths[spw_no,chan_no])

def compute_vis_grid_indicies(should_average_spw_channels,should_average_all,data,enabled_channels,channels_to_image):
  channel_grid_index = np.zeros([data._no_spw*data._no_channels],dtype=np.intp) #stores the index of the grid this channel should be saved to (usage: image cubes)
  cube_chan_dim_size = 0
  
  if should_average_spw_channels:
    cube_chan_dim_size += 1 #at least one channel, so at least one spw
    current_spw = channels_to_image[0] / data._no_channels
    current_grid = 0
    for c in range(0,len(enabled_channels)):
      channel_grid_index[c] = current_grid
      if enabled_channels[c] and (c / data._no_channels) > current_spw:
	cube_chan_dim_size += 1
	current_grid += 1
	current_spw = c / data._no_channels
  elif len(channels_to_image) > 1 and not should_average_all: #grid individual channels
    current_grid = 0
    for c in range(0,len(enabled_channels)):
      channel_grid_index[c] = current_grid
      if enabled_channels[c]:
	current_grid += 1
	cube_chan_dim_size += 1	
  else:
    cube_chan_dim_size = 1
  return (channel_grid_index,cube_chan_dim_size)

def compute_sampling_function_grid_indicies(data,channels_to_image):
  sampling_function_channel_grid_index = np.zeros([data._no_spw*data._no_channels],dtype=np.intp)
  sampling_function_channel_count = len(channels_to_image)
  current_grid = 0
  for c in range(0,len(enabled_channels)):
    sampling_function_channel_grid_index[c] = current_grid
    if enabled_channels[c]:
      current_grid += 1
  return sampling_function_channel_grid_index