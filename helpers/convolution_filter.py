'''
Created on Mar 20, 2014

@author: bhugo
'''
import numpy as np
import math
import fft_utils
import base_types
import matplotlib.pyplot as plt

DEBUG_KERNEL_ON = False

class convolution_filter(object):
  '''
  classdocs
  '''

  def gausian(self,x,grid_max_x):
    dx = 1.0 / grid_max_x
    return (1.0/(0.2432*dx*math.sqrt(math.pi))*math.exp(-(x/0.2432*dx)**2))
	   
      
  def gausian_fourier(self,x,grid_max_x):
    dx = 1.0 / grid_max_x  
    return (math.exp(-(math.pi*0.2432*dx*x)**2))
  '''
  The [Modified] Keiser Bessel Function
  See:
  Jackson, John I., et al. "Selection of a convolution function for Fourier inversion 
  using gridding [computerised tomography application]." Medical Imaging, IEEE Transactions on 10.3 (1991): 473-478.
  
  This reference also have a few points on the topic:
  Prabhu, K. M. M. Window Functions and Their Applications in Signal Processing. CRC Press, 2013.
  '''

  def set_keiser_bessel_beta(self,conv_support):
    self.beta = 1.9980 if conv_support == 1 else 1.678933333*conv_support-0.959033333 # from Jackson et al. this line seems to fit the tabled betas very well
  
  def modified_zero_order_bessel_first_kind(self,z):
    max_k = 15 
    result = 0
    for k in range(0,max_k):
      result += ((z/2.0)**(2*k))/(math.factorial(k)**2)
    return result

  def keiser_bessel(self,x,w):
    return self.modified_zero_order_bessel_first_kind(self.beta*math.sqrt(1-(2.0*x/w)**2))*(1.0/w)
  
  def keiser_bessel_fourier(self,x,w):
    sqrt_term_sq = (math.pi*w*x)**2 - self.beta**2
    sqrt_term = np.sqrt(sqrt_term_sq) if sqrt_term_sq > 0 else 0.00000001 #ignore imaginary component since the image is supposed to be real anyway
    return np.sin(sqrt_term)/sqrt_term

  '''
  Constructor
  '''
  def __init__(self, convolution_fir_support, oversampling_factor, grid_size_l, grid_size_m, function_to_use="gausian"):
    convolution_func = { "gausian" : self.gausian , "keiser bessel" : self.keiser_bessel }
    detaper_func = { "gausian" : self.gausian_fourier , "keiser bessel" : self.keiser_bessel_fourier }
    self.set_keiser_bessel_beta(convolution_fir_support)
    
    print("CREATING CONVOLUTION FILTER... "),
    convolution_size = convolution_fir_support * oversampling_factor
    convolution_centre = convolution_size / 2
    
    conv_FIR_dim = np.empty([convolution_size],dtype=base_types.fir_type)
    for x in range(0,convolution_size):
      index = x
      conv_FIR_dim[index] = convolution_func[function_to_use]((x - convolution_centre)/float(oversampling_factor),convolution_fir_support)
    
    self._conv_FIR = np.outer(conv_FIR_dim,conv_FIR_dim) # the tensor product is probably the fastest way to get this matrix out with python
    self._conv_FIR /= np.sum(self._conv_FIR) #the convolution function should integrate to unity, so that the whole image doesn't change when we change the oversampling
    if DEBUG_KERNEL_ON:
      plt.figure()
      plt.title("Convolution Kernel")
      plt.imshow(self._conv_FIR)
      plt.show(False)
    print " <DONE>"
    
    print("CREATING DETAPERER... "),
    detaper_centre_l = grid_size_l / 2
    detaper_centre_m = grid_size_m / 2
	
    F_detaper_l = np.zeros([grid_size_l],dtype=base_types.detaper_type)
    for li in range(0,grid_size_l):
      F_detaper_l[li] = detaper_func[function_to_use]((li-detaper_centre_l)/float(grid_size_l),convolution_fir_support)
    F_detaper_m = np.zeros([grid_size_m],dtype=base_types.detaper_type)
    for mi in range(0,grid_size_m):
      F_detaper_m[mi] = detaper_func[function_to_use]((mi-detaper_centre_m)/float(grid_size_m),convolution_fir_support)
      
    self._F_detaper = np.outer(F_detaper_l,F_detaper_m) # the tensor product is probably the fastest way to get this matrix out with python
    if DEBUG_KERNEL_ON:
      plt.figure()
      plt.title("AA Taper")
      plt.imshow(self._F_detaper)
      plt.show(True)
      #exit()
    print " <DONE>"
