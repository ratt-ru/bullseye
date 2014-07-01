'''
Created on Mar 20, 2014

@author: bhugo
'''
import numpy as np
import math
import fft_utils

def gausian(u,v,grid_max_u,grid_max_v):
  du = 1.0 / grid_max_u
  dv = 1.0 / grid_max_v  
    
  return ((1.0/(0.750*du*math.sqrt(math.pi))*math.exp(-(u/0.750*du)**2)) *
	  (1.0/(0.750*dv*math.sqrt(math.pi))*math.exp(-(v/0.750*dv)**2)))
      
def gausian_fourier(l,m,grid_max_u,grid_max_v):
  du = 1.0 / grid_max_u
  dv = 1.0 / grid_max_v  
    
  return ((math.exp(-(math.pi*0.750*du*l)**2)) *
	  (math.exp(-(math.pi*0.750*dv*m)**2)))

class convolution_filter(object):
    '''
    classdocs
    '''
    def __init__(self, convolution_fir_support_u,convolution_fir_support_v, oversampling_factor, grid_size_l, grid_size_m):
        '''
        Constructor
        '''
        print("CREATING CONVOLUTION FILTER... "),
	convolution_size_u = convolution_fir_support_u * oversampling_factor
	convolution_size_v = convolution_fir_support_v * oversampling_factor
        convolution_centre_u = convolution_size_u / 2
        convolution_centre_v = convolution_size_v / 2
	self._conv_FIR = np.zeros([convolution_size_u,convolution_size_v],dtype=np.float32)
        for vi in range(0,convolution_size_v):
            for ui in range(0,convolution_size_u):
                self._conv_FIR[ui,vi] = gausian((ui - convolution_centre_u)/float(convolution_size_u)*convolution_fir_support_u,
                                                (vi - convolution_centre_v)/float(convolution_size_v)*convolution_fir_support_v,
                                                grid_size_l,grid_size_m)
        print " <DONE>"
        
        print("CREATING DETAPERER... "),
	
        detaper_centre_l = grid_size_l / 2
        detaper_centre_m = grid_size_m / 2
	
	self._F_detaper = np.zeros([grid_size_l,grid_size_m],dtype=np.float32)
	for mi in range(0,grid_size_m):
	  for li in range(0,grid_size_l):
            self._F_detaper[li,mi] = gausian_fourier((li - detaper_centre_l)/float(detaper_centre_l)*convolution_fir_support_u, 
						     (mi - detaper_centre_m)/float(detaper_centre_m)*convolution_fir_support_v,
						     grid_size_l,grid_size_m)
        print " <DONE>"
