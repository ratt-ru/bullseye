'''
Created on Mar 20, 2014

@author: bhugo
'''
import numpy as np
import math
class convolution_filter(object):
    '''
    classdocs
    '''
    def __init__(self, convolution_fir_support, oversampling_factor):
        '''
        Constructor
        '''
        print("CREATING CONVOLUTION FILTER... "),
	convolution_size = convolution_fir_support * oversampling_factor
        convolution_centre = convolution_size / 2
	self._conv_FIR = np.zeros([convolution_size,convolution_size])
        for vi in range(0,convolution_size):
            for ui in range(0,convolution_size):
                self._conv_FIR[vi,ui] = self.__gausian_sinc((ui-convolution_centre)*(convolution_fir_support/float(convolution_size)),
                                                           (vi-convolution_centre)*(convolution_fir_support/float(convolution_size)),
                                                           convolution_fir_support,convolution_fir_support)
        print " <DONE>"
    def __sinc2D(self,x,y):
    	px = math.pi*x+0.000001
    	py = math.pi*y+0.000001
    	return (math.sin(px)*math.sin(py))/(px*py)

    def __gausian_sinc(self,u,v,grid_max_u,grid_max_v):
    	u = u+0.000001
    	v = v+0.000001
    	du = 1.0 / grid_max_u
    	dv = 1.0 / grid_max_v  
    
    	return ((math.sin(np.pi*u/(1.55*du))/(np.pi*u))*math.exp(-(u/(2.52*du))**2)*
            	(math.sin(np.pi*v/(1.55*dv))/(np.pi*v))*math.exp(-(v/(2.52*dv))**2))

    def __rect_func(self,u,v,grid_max_u,grid_max_v):
    	du = 1.0 / grid_max_u
    	dv = 1.0 / grid_max_v
    	return ((1/du)*(1.0 if abs(u/du) <= 0.5 else 0.0)) * ((1/dv)*(1.0 if abs(v/dv) <= 0.5 else 0.0))
