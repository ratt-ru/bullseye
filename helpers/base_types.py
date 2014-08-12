import numpy as np
import ctypes
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
global uvw_type
uvw_type = np.float32
global uvw_ctypes_convert_type
uvw_ctypes_convert_type = ctypes.c_float
global reference_wavelength_type
reference_wavelength_type = np.float32
global weight_type
weight_type = np.float32