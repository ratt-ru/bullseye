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
import numpy as np
import ctypes
global grid_type
grid_type = None
global psf_type
psf_type = None
global fir_type
fir_type = None
global w_fir_type
w_fir_type = None
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
    global w_fir_type
    w_fir_type = np.complex64
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
    global w_fir_type
    w_fir_type = np.complex128
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