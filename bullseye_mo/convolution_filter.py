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
from scipy import special as sp
from pyrap.quanta import quantity

class convolution_filter(object):
  '''
  classdocs
  '''
  
  '''
  Sinc function with discontinuity removed
  '''
  def sinc(self,x):
    param = x * np.pi
    if x == 0:
      return 1 #remove discontinuity
    else:
      return np.sin(param)/param
    
  '''
  Each of these windowsing functions should take an 
  array of indicies, the width of the full 
  support and oversampling factor. Each should return a tap
  '''
  def unity(self,x,W,oversample):
    return np.ones(len(x))
  
  def kb(self,x,W,oversample):
    beta = (7.43 - 2.39) / (5.0-2.0) * W - 0.366833
    sqrt_inner = 1 - (x/float(W))**2
    normFactor = oversample * sp.i0(W)
    return sp.i0(W * np.sqrt(sqrt_inner)) * normFactor
  
  def hamming(self,x,W,oversample):
    return np.hamming(W*oversample)
  
  '''
  Constructor
  convolution_fir_support is the half support of the filter
  oversampling_factor should be reasonably large to account for interpolation error
  function_to_use can either be sinc, kb, or hamming (these will all be seperable filters and the output will be a real-valued 1D filter if w projection is turned off
  if use_w_projection == True then
    - specify wplanes (usually this should be enough to ensure the seperation between planes << 1
    - specify npix_l and npix_m: the number of image pixels in l and m
    - specify cell_l and cell_m: the cell sizes in arcsec
    - w_max must be measured in terms of the lowest lambda and the maximum w in the measurement
    - the output now will be a stack of inseperable 2D complex filters with the AA filter already incorporated
  '''
  def __init__(self, convolution_fir_support, oversampling_factor, function_to_use="sinc", use_w_projection = False, 
	       wplanes = 1, npix_l=1,npix_m=1,celll=1,celllm=1,w_max=1):
    convolution_func = { "sinc" : self.unity, "kb" : self.kb, "hamming" : self.hamming }

    print("CREATING CONVOLUTION FILTERS... ")
    convolution_full_support = convolution_fir_support * 2 + 1 + 2 #see convolution_policies.h for more details
    convolution_size = (convolution_full_support) * oversampling_factor
    convolution_centre = convolution_size / 2
    AA = np.empty([convolution_size],dtype=base_types.fir_type)
    for x in range(0,convolution_size):
      AA[x] = self.sinc((x - convolution_centre)/float(oversampling_factor))
    x = ((np.arange(0, convolution_size) - convolution_centre))/float(oversampling_factor)
    AA *= convolution_func[function_to_use](x,convolution_full_support,oversampling_factor)
    AA /= np.sum(AA) #normalize to unity
    
    if not use_w_projection:
      self._conv_FIR = AA
    else:
      AA_2D = np.outer(AA,AA) #the outer product is probably the fastest way to generate the 2D anti-aliasing filter from our 1D version
      F_AA_2D = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(AA_2D)))
      '''
      image parameters
      '''
      celll = quantity(celll,"arcsec").get_value("rad")
      cellm = quantity(celll,"arcsec").get_value("rad")
      theta_max = npix_l * celll
      phi_max = npix_m * cellm
      image_diam = np.sqrt(theta_max**2 + phi_max**2)
      #Recommended support as per Tasse (Applying full polarization A-projection to very wide field of view instruments: an imager for LOFAR)
      recommended_half_support = int(np.ceil(((4 * np.pi * w_max * image_diam**2) / np.sqrt(2 - image_diam**2)) * 0.5))
      print "The recommended half support region for the convolution kernel is", recommended_half_support
      '''
      generate the filter over theta and phi where
      l = cos(theta)*cos(phi)
      m = sin(theta)*sin(phi)
      n = sqrt(1-l**2-m**2)
      l and m are the direction cosines relative to the phase centre (let the image touch the celestial sphere at this point)
      '''
      theta = np.linspace(0,theta_max,convolution_size) - theta_max * 0.5
      phi = np.linspace(0,phi_max,convolution_size) - phi_max * 0.5
      thetag,phig = np.meshgrid(theta,phi)
      n = np.sqrt(1-(np.cos(thetag)*np.cos(phig))**2-(np.sin(thetag)*np.sin(phig))**2)
      '''
      The W term in the ME is exp(2*pi*1.0j*w*(n-1))
      Therefore we'll get the cos and sine fringes, which we can then fourier transform
      '''
      W_kernel = (np.exp(2*np.pi*1.0j*(n-1)).astype(base_types.w_fir_type))
      W_kernels = np.empty([wplanes,
                      convolution_size,
                      convolution_size],dtype=base_types.w_fir_type)
      '''
      We want w*(n-1), so raise to the power of w. 
      Then by convolution theorem multiply in F_AA_2D so that we end up convolving with AA_2D 
      '''
      plane_step = w_max / float(wplanes)
      for w in range(0,wplanes):
	W_kernels[w,:,:] = (W_kernel ** ((w+1)*plane_step)) * F_AA_2D
      self._conv_FIR = W_kernels
    print " CONVOLUTION FILTERS CREATED"
