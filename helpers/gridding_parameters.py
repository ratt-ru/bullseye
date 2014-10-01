import base_types
from ctypes import *
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
  ("sampling_function_buffer",c_void_p),
  ("sampling_function_channel_grid_indicies",c_void_p),
  ("sampling_function_channel_count",c_size_t)
]
