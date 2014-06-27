'''
Created on Mar 22, 2014

@author: bhugo
'''
import scipy.fftpack
import numpy as np

F2=scipy.fftpack.fft2
iF2=scipy.fftpack.ifft2
Fs=scipy.fftpack.fftshift
iFs=scipy.fftpack.ifftshift

def fft2(A):
    FA= Fs(F2(iFs(A)))/(np.float64(A.size))
    return FA
 
def ifft2(A):    
    FA=Fs(iF2(iFs(A)))*np.float64(A.size)
    return FA
