#pragma once
#include <boost/iterator/iterator_concepts.hpp>

template <typename T> struct basic_complex { 
  T _real,_imag; 
  __device__ __host__ basic_complex(T real = 0, T imag = 0):_real(real),_imag(imag) {}
  __device__ __host__ basic_complex<T>& operator+=(const basic_complex<T> & rhs){
    _real += rhs._real;
    _imag += rhs._imag;
    return *this;
  }
  __device__ __host__ basic_complex<T> & operator= (const basic_complex<T> & rhs){
    _real = rhs._real;
    _imag = rhs._imag;
    return *this;
  }
  __device__ __host__ basic_complex<T> operator*(const T scalar){
    return basic_complex<T>(_real * scalar, _imag * scalar);
  }
  __device__ __host__ basic_complex<T> operator*(const basic_complex<T> & rhs) {
    return basic_complex<T>(_real * rhs._real - _imag * rhs._imag, _real * rhs._imag + _imag * rhs._real);
  }
  __device__ __host__ basic_complex<T>& operator*=(const basic_complex<T> & rhs) {
    T n_real = _real * rhs._real - _imag * rhs._imag;
    T n_imag = _real * rhs._imag + _imag * rhs._real;
    _real = n_real;
    _imag = n_imag;
    return *this;
  }
};