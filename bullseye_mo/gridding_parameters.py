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
import bullseye_mo.base_types as base_types
from ctypes import *

if base_types.uvw_ctypes_convert_type == None:
  raise Exception("Please import base_types.py first and select a precision mode before importing gridding_parameters.py")
class gridding_parameters(Structure):
  pass
gridding_parameters._fields_ = [
  #Mandatory data necessary for gridding:
  ("visibilities",c_void_p),
  ("uvw_coords",c_void_p),
  ("reference_wavelengths",c_void_p),
  ("visibility_weights",c_void_p),
  ("flags",c_void_p),
  ("flagged_rows",c_void_p),
  ("field_array",c_void_p),
  ("spw_index_array",c_void_p),
  ("imaging_field",c_uint), #mandatory: used to seperate different pointings in the MS 2.0 specification
  #Mandatory count fields necessary for gridding:
  ("baseline_count",c_size_t),
  ("row_count",c_size_t),
  ("chunk_max_row_count",c_size_t),
  ("channel_count",c_size_t),
  ("number_of_polarization_terms",c_size_t),
  ("number_of_polarization_terms_being_gridded",c_size_t),
  ("spw_count",c_size_t),
  ("no_timestamps_read",c_size_t),
  #Mandatory image scaling fields necessary for scaling the IFFT correctly
  ("nx",c_size_t),
  ("ny",c_size_t),
  ("cell_size_x",base_types.uvw_ctypes_convert_type),
  ("cell_size_y",base_types.uvw_ctypes_convert_type),
  #Fields in use for specifying externally computed convolution function
  ("conv",c_void_p),
  ("conv_support",c_size_t),
  ("conv_oversample",c_size_t),
  #Correlation index specifier for gridding a single stokes/correlation term
  ("polarization_index",c_size_t),
  ("second_polarization_index",c_size_t),
  #Preallocated gridding buffer
  ("output_buffer",c_void_p),
  #Faceting information
  ("phase_centre_ra",base_types.uvw_ctypes_convert_type),
  ("phase_centre_dec",base_types.uvw_ctypes_convert_type),
  ("facet_centres",c_void_p),
  ("num_facet_centres",c_size_t),
  #Fields required to specify jones facet_4_cor_corrections
  ("jones_terms",c_void_p),
  ("should_invert_jones_terms",c_bool),
  ("antenna_1_ids",c_void_p),
  ("antenna_2_ids",c_void_p),
  ("timestamp_ids",c_void_p),
  ("antenna_count",c_size_t),
  #Channel selection and averaging
  ("enabled_channels",c_void_p),
  ("channel_grid_indicies",c_void_p),
  ("cube_channel_dim_size",c_size_t),
  #Sampling function
  ("should_grid_sampling_function",c_bool),
  ("sampling_function_buffer",c_void_p),
  ("sampling_function_channel_grid_indicies",c_void_p),
  ("sampling_function_channel_count",c_size_t),
  #Finalization steps
  ("is_final_data_chunk",c_bool),
  #w-projection related terms
  ("wplanes",c_size_t),
  ("wmax_est",base_types.uvw_ctypes_convert_type),
  #baseline indexes needed for Romeins distribution strategy
  ("baseline_starting_indexes",c_void_p), #this has to be n(n-1)/2 + n + 1 long because we need to compute the length of the last baseline
  #The following will be allocated and released in the C libraries
  ("antenna_jones_starting_indexes",c_void_p), #this has to be n + 1 long because we need to be able to compute the number of jones terms at the last antenna
  ("jones_time_indicies_per_antenna",c_void_p), #this will be the same length as the repacked jones matrix array
  ("normalization_terms",c_void_p) #this has to be threads_bins x #facets x #channel_accumulation_grids x #polarization_being_gridded
]
