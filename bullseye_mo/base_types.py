import numpy as np
import ctypes
global grid_type
grid_type = None
global psf_type
psf_type = None
global fir_type
fir_type = None
global detaper_type
detaper_type = None
global visibility_type
visibility_type = None
global visibility_component_type
visibility_component_type = None
global uvw_type
uvw_type = None
global uvw_ctypes_convert_type
uvw_ctypes_convert_type = None
global reference_wavelength_type
reference_wavelength_type = None
global weight_type
weight_type = None

def force_precision(precision):
  if precision == 'single':
    global grid_type
    grid_type = np.complex64
    global psf_type
    psf_type = grid_type
    global fir_type
    fir_type = np.float32
    global detaper_type
    detaper_type = np.float32
    global visibility_type
    visibility_type = np.complex64
    global visibility_component_type
    visibility_component_type = np.float32
    global uvw_type
    uvw_type = np.float32
    global uvw_ctypes_convert_type
    uvw_ctypes_convert_type = ctypes.c_float if uvw_type == np.float32 else ctypes.c_double
    global reference_wavelength_type
    reference_wavelength_type = np.float32
    global weight_type
    weight_type = np.float32
  elif precision == 'double':
    global grid_type
    grid_type = np.complex128
    global psf_type
    psf_type = grid_type
    global fir_type
    fir_type = np.float64
    global detaper_type
    detaper_type = np.float64
    global visibility_type
    visibility_type = np.complex128
    global visibility_component_type
    visibility_component_type = np.float64
    global uvw_type
    uvw_type = np.float64
    global uvw_ctypes_convert_type
    uvw_ctypes_convert_type = ctypes.c_float if uvw_type == np.float32 else ctypes.c_double
    global reference_wavelength_type
    reference_wavelength_type = np.float64
    global weight_type
    weight_type = np.float64
  else:
    raise Exception("base_types don't understand your argument.. must be 'single'/'double'")