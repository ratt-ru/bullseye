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
import ctypes
import os

def load_library(architecture,precision):
  if architecture not in ['CPU','GPU']:
    raise Exception("Invalid architecture, only CPU or GPU allowed")
  if precision not in ['single','double']:
    raise Exception("Invalid precision mode selected, only single or double allowed")
  mod_path = os.path.dirname(__file__)
  libimaging = None
  if architecture == 'CPU':
    if precision == 'single':
      libimaging = ctypes.pydll.LoadLibrary("%s/cbuild/new_multithreaded_approach/single/libalt_imaging32.so" % mod_path)
    else:
      libimaging = ctypes.pydll.LoadLibrary("%s/cbuild/new_multithreaded_approach/double/libalt_imaging64.so" % mod_path)
  elif architecture == 'GPU':
    if precision == 'single':
      libimaging = ctypes.pydll.LoadLibrary("%s/cbuild/gpu_algorithm/single/libgpu_imaging32.so" % mod_path)
    else:
      libimaging = ctypes.pydll.LoadLibrary("%s/cbuild/gpu_algorithm/double/libgpu_imaging64.so" % mod_path)
  return libimaging
