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
from scipy import signal as sig
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
    return np.hamming(W + (W-1)*(oversample-1))
  
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
	       wplanes = 1, npix_l=1,npix_m=1,celll=1,cellm=1,w_max=1,ra_0=0,dec_0=0):
    convolution_func = { "sinc" : self.unity, "kb" : self.kb, "hamming" : self.hamming }

    print("CREATING CONVOLUTION FILTERS... ")
    full_support = convolution_fir_support * 2 + 1
    convolution_full_support = full_support + 2 #padding by oversample number of samples at both ends
    #oversampling number of jumps in between each major tap (including the 2 padding taps)
    convolution_size = convolution_full_support + ((convolution_full_support - 1) * (oversampling_factor - 1)) #filter array dimension size
    convolution_centre = convolution_size // 2
    oversample_image_space = (oversampling_factor / (float(convolution_centre)))**-1
    sup_image_space_scale = convolution_fir_support / float(oversample_image_space)
    
    x = np.arange(0, convolution_centre+1)
    x = np.hstack((-x[::-1],x[1:]))
    AA = np.sinc(x/float(oversampling_factor))
    AA *= convolution_func[function_to_use](x/float(oversampling_factor),convolution_full_support,oversampling_factor)
    AA /= np.sum(AA) #normalize to unity
    
    #compute number of facets required if no w-projection is applied
    phase_error_threshold = 0.5 # this should be much less than 1 for the 2D FFT to be valid
    celll = quantity(celll,"arcsec").get_value("rad")
    cellm = quantity(cellm,"arcsec").get_value("rad")
    ra_max = npix_l * celll
    dec_max = npix_m * cellm
    dec_0 = quantity(dec_0,"arcsec").get_value("rad")
    ra_0 = quantity(ra_0,"arcsec").get_value("rad")
    epsilon_max = abs(np.sin(dec_max*0.5 + dec_0)*np.sin(dec_0) + 
		      np.cos(dec_max*0.5 + dec_0)*np.cos(dec_0)*np.cos(ra_max*0.5) - 
		      (phase_error_threshold)/(2*np.pi*w_max)) # same as n - ni for small angles
    num_facets_needed = np.ceil(np.sqrt(max(celll*npix_l,cellm*npix_m) / (2*np.arccos(epsilon_max)))*2)
    print "The recommended number of facets (using no w-projection) along each direction of the image is approximately", num_facets_needed
    if not use_w_projection:
      self._conv_FIR = AA.astype(dtype=base_types.fir_type) #Seperable 1D AA filter (correct policy must be called in the gridder code
      print "WARNING: DISABLING W-PROJECTION"
    else:
      AA_2D = np.outer(AA,AA) #the outer product is probably the fastest way to generate the 2D anti-aliasing filter from our 1D version
      F_AA_2D = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(AA_2D))) / (convolution_size**2)
      image_diam = np.sqrt(ra_max**2 + dec_max**2)
      #Recommended support as per Tasse (Applying full polarization A-projection to very wide field of view instruments: an imager for LOFAR)
      recommended_half_support_image = int(np.ceil(((4 * np.pi * w_max * image_diam**2) / np.sqrt(2 - image_diam**2)) * 0.5))
      recommended_half_support = int(np.ceil(1 / float(oversampling_factor) * recommended_half_support_image))
      print "The recommended half support region for the convolution kernel in image space is (Nyquest):", recommended_half_support_image
      print "The recommended half support region for the convolution kernel in fourier space at specified oversampling rate is (recommended input to imager):", recommended_half_support
      epsilon_max = abs(np.sin(dec_max*0.5 + dec_0) * np.sin(dec_0) + np.cos(dec_max*0.5 + dec_0)*np.cos(dec_0)*np.cos(ra_max*0.5) - 1) # same as n - 1
      recommended_nplanes = int(np.ceil(2*np.pi*w_max*epsilon_max/phase_error_threshold)) #this is half the number of planes required... the other half need not be calculated because we grid the conjugate vis
      print "The recommended number of w-planes (using no faceting) are", recommended_nplanes, "to obtain a maximum seperation of", phase_error_threshold, "between planes"
      '''
      generate the filter over theta and phi where
      l = cos(dec)*sin(ra-ra0)
      m = sin(dec)*cos(dec0)-cos(dec)*sin(dec0)*cos(ra-ra0)
      n = sqrt(1-l**2-m**2)
      '''
      dec = x / float(oversample_image_space) / float(sup_image_space_scale) * (dec_max * 0.5) 
      ra = x / float(oversample_image_space) / float(sup_image_space_scale) * (ra_max * 0.5) 

      decg,rag = np.meshgrid(dec,ra,indexing='ij')

      l = np.cos(decg+dec_0)*np.sin(rag)
      m = np.sin(decg+dec_0)*np.cos(dec_0)-np.cos(decg+dec_0)*np.sin(dec_0)*np.cos(rag)
      n = np.sqrt(1-l**2-m**2)
      
      '''
      The W term in the ME is exp(2*pi*1.0j*w*(n-1))
      Therefore we'll get the cos and sine fringes, which we can then fourier transform
      '''
      W_bar_kernels = np.zeros([wplanes,
				convolution_size,
				convolution_size],dtype=base_types.w_fir_type)
      plane_step = w_max / float(wplanes - 1)
      for w in range(0,wplanes):  
	  W_kernel = F_AA_2D * np.exp(-2*np.pi*1.0j*(n-1)* (w*plane_step)) / n
	  W_bar_kernels[w,:,:] = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(W_kernel))) / (convolution_size**2)
	  #plt.figure()
	  #plt.plot(np.real(W_bar_kernels[w,W_bar_kernels.shape[1]//2,:]))
	  #plt.show()
      #exit(1)
      self._conv_FIR = W_bar_kernels.astype(base_types.w_fir_type)
    print "CONVOLUTION FILTERS CREATED"
