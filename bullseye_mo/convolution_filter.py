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
import bullseye_mo.base_types as base_types
import matplotlib.pyplot as plt

DEBUG_KERNEL_ON = False

class convolution_filter(object):
  '''
  classdocs
  '''

  def sinc(self,x,grid_max_x):
    param = x * np.pi
    if x == 0:
      return 1 #remove discontinuity
    else:
      return np.sin(param)/param
  '''
  Constructor
  '''
  def __init__(self, convolution_fir_support, oversampling_factor, grid_size_l, grid_size_m, function_to_use="sinc"):
    convolution_func = { "sinc" : self.sinc }

    print("CREATING CONVOLUTION FILTER... "),
    convolution_full_support = convolution_fir_support * 2 + 1 + 2 #see convolution_policies.h for more details
    convolution_size = (convolution_full_support) * oversampling_factor
    convolution_centre = convolution_size / 2

    self._conv_FIR = np.empty([convolution_size],dtype=base_types.fir_type)
    for x in range(0,convolution_size):
      self._conv_FIR[x] = convolution_func[function_to_use]((x - convolution_centre)/float(oversampling_factor),convolution_fir_support)

    self._conv_FIR / np.sum(self._conv_FIR) #normalize to unity
    #coalseced_FIR = np.zeros([convolution_size],dtype=base_types.fir_type)
    #for s in range(0,convolution_full_support):
      #coalseced_FIR[s*convolution_full_support:(s+1)*convolution_full_support] = self._conv_FIR[s::oversampling_factor]
    #self._conv_FIR = coalseced_FIR

    if DEBUG_KERNEL_ON:
      plt.figure()
      plt.title("Convolution Kernel")
      plt.plot(self._conv_FIR)
      plt.show(True)
    print " <DONE>"
