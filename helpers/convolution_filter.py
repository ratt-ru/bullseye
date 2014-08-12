'''
Created on Mar 20, 2014

@author: bhugo
'''
import numpy as np
import math
import fft_utils
import base_types
class convolution_filter(object):
  '''
  classdocs
  '''

  def gausian(self,u,v,grid_max_u,grid_max_v):
    du = 1.0 / grid_max_u
    dv = 1.0 / grid_max_v  
    
    return ((1.0/(0.2432*du*math.sqrt(math.pi))*math.exp(-(u/0.2432*du)**2)) *
	    (1.0/(0.2432*dv*math.sqrt(math.pi))*math.exp(-(v/0.2432*dv)**2)))
      
  def gausian_fourier(self,l,m,grid_max_u,grid_max_v):
    du = 1.0 / grid_max_u
    dv = 1.0 / grid_max_v  
    
    return ((math.exp(-(math.pi*0.2432*du*l)**2)) *
	    (math.exp(-(math.pi*0.2432*dv*m)**2)))
  '''
  The [Modified] Keiser Bessel Function
  See:
  Jackson, John I., et al. "Selection of a convolution function for Fourier inversion 
  using gridding [computerised tomography application]." Medical Imaging, IEEE Transactions on 10.3 (1991): 473-478.
  
  This reference also have a few points on the topic:
  Prabhu, K. M. M. Window Functions and Their Applications in Signal Processing. CRC Press, 2013.
  '''

  def set_keiser_bessel_beta(self,conv_support_u,conv_support_v):
    self.beta_u = 1.678933333*conv_support_u-0.959033333 # from Jackson et al. this line seems to fit the tabled betas very well
    self.beta_v = 1.678933333*conv_support_v-0.959033333 # from Jackson et al. this line seems to fit the tabled betas very well
  
  def modified_zero_order_bessel_first_kind(self,z):
    max_k = 15 
    result = 0
    for k in range(0,max_k):
      result += ((z/2.0)**(2*k))/(math.factorial(k)**2)
    return result

  def keiser_bessel(self,u,v,grid_max_u,grid_max_v):
    return (self.modified_zero_order_bessel_first_kind(self.beta_u*math.sqrt(1-(2.0*u)**2)) *
	    self.modified_zero_order_bessel_first_kind(self.beta_v*math.sqrt(1-(2.0*v)**2)))
  
  def keiser_bessel_fourier(self,l,m,grid_max_u,grid_max_v):
    sqrt_term_l_sq = (math.pi*l)**2 - self.beta_u**2 #assuming W = 1
    sqrt_term_m_sq = (math.pi*m)**2 - self.beta_v**2 #assuming W = 1
    sqrt_term_l = np.sqrt(sqrt_term_l_sq) if sqrt_term_l_sq > 0 else 0.00000001 #ignore imaginary component since the image is supposed to be real anyway
    sqrt_term_m = np.sqrt(sqrt_term_m_sq) if sqrt_term_m_sq > 0 else 0.00000001 #ignore imaginary component since the image is supposed to be real anyway
    return np.sin(sqrt_term_l)/sqrt_term_l*math.sin(sqrt_term_m)/sqrt_term_m

  '''
  Constructor
  '''
  def __init__(self, convolution_fir_support_u,convolution_fir_support_v, oversampling_factor, grid_size_l, grid_size_m, function_to_use="gausian"):
    convolution_func = { "gausian" : self.gausian , "keiser bessel" : self.keiser_bessel }
    detaper_func = { "gausian" : self.gausian_fourier , "keiser bessel" : self.keiser_bessel_fourier }
    self.set_keiser_bessel_beta(convolution_fir_support_u,convolution_fir_support_v)
    
    print("CREATING CONVOLUTION FILTER... "),
    convolution_size_u = convolution_fir_support_u * oversampling_factor
    convolution_size_v = convolution_fir_support_v * oversampling_factor
    convolution_centre_u = convolution_size_u / 2
    convolution_centre_v = convolution_size_v / 2
    self._conv_FIR = np.zeros([convolution_size_u,convolution_size_v],dtype=base_types.fir_type)
    for vi in range(0,convolution_size_v):
      for ui in range(0,convolution_size_u):
	self._conv_FIR[ui,vi] = convolution_func[function_to_use]((ui - convolution_centre_u)/float(convolution_size_u),
								  (vi - convolution_centre_v)/float(convolution_size_v),
								  grid_size_l,grid_size_m)
    print " <DONE>"
        
    print("CREATING DETAPERER... "),
	
    detaper_centre_l = grid_size_l / 2
    detaper_centre_m = grid_size_m / 2
	
    self._F_detaper = np.zeros([grid_size_l,grid_size_m],dtype=base_types.detaper_type)
    for mi in range(0,grid_size_m):
      for li in range(0,grid_size_l):
	self._F_detaper[li,mi] = detaper_func[function_to_use]((li-detaper_centre_l)/float(detaper_centre_l), 
							       (mi-detaper_centre_m)/float(detaper_centre_m),
							       grid_size_l,grid_size_m)
    print " <DONE>"
