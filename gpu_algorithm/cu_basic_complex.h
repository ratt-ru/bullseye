#pragma once

template <typename T> struct basic_complex { 
  T _real,_imag; 
  __device__ __host__ basic_complex(T real = 0, T imag = 0):_real(real),_imag(imag) {}
};