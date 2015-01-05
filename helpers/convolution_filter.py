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
    convolution_size = (convolution_fir_support * 2 + 1 + 2) * oversampling_factor #see convolution_policies.h for more details
    convolution_centre = convolution_size / 2
    
    self._conv_FIR = np.empty([convolution_size],dtype=base_types.fir_type)
    for x in range(0,convolution_size):
      self._conv_FIR[x] = convolution_func[function_to_use]((x - convolution_centre)/float(oversampling_factor),convolution_fir_support)
    
    if DEBUG_KERNEL_ON:
      plt.figure()
      plt.title("Convolution Kernel")
      plt.plot(self._conv_FIR)
      plt.show(True)
    print " <DONE>"
