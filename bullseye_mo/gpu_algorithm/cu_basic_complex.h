/********************************************************************************************
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
********************************************************************************************/
#pragma once
#include <boost/iterator/iterator_concepts.hpp>

template <typename T> struct basic_complex { 
  T _real,_imag; 
  __device__ __host__ basic_complex(T real = 0, T imag = 0):_real(real),_imag(imag) {}
  __device__ __host__ basic_complex<T> operator+(const basic_complex<T> & rhs){
    return basic_complex<T>(_real + rhs._real,_imag + rhs._imag);
  }
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
  __device__ __host__ bool operator== (const basic_complex<T> & rhs){
    return _real == rhs._real && _imag == rhs._imag;
  }
  __device__ __host__ basic_complex<T> operator*(const T scalar){
    return basic_complex<T>(_real * scalar, _imag * scalar);
  }
  __device__ __host__ basic_complex<T> operator-(){
    return basic_complex<T>(-_real, -_imag);
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
  __device__ __host__ basic_complex<T> operator/(const basic_complex<T> & denum) {
    //multiply with the transpose of the denumerator at the bottom and top. The new denumerator is guarenteed to be real so we can divide through element-wise
    basic_complex<T> conj_denumerator(denum._real,-denum._imag);
    T frac_after_conj_mul = (denum * conj_denumerator)._real;
    basic_complex<T> numerator = (*this * conj_denumerator) * (1 / frac_after_conj_mul);
    return numerator;
  }
};
template <typename T>
__device__ __host__ basic_complex<T> operator*(const basic_complex<T> & lhs,const basic_complex<T> & rhs) {
    return basic_complex<T>(lhs._real * rhs._real - lhs._imag * rhs._imag, lhs._real * rhs._imag + lhs._imag * rhs._real);
}
template <typename T>
__device__ __host__ basic_complex<T> operator-(const basic_complex<T> & lhs,const basic_complex<T> & rhs) {
    return basic_complex<T>(lhs._real - rhs._real, lhs._imag - rhs._imag);
}