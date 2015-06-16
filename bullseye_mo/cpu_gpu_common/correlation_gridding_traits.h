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
#include <x86intrin.h>
#include "cu_common.h"
#include "cu_vec.h"
#include "cu_basic_complex.h"
#include "jones_2x2.h"

namespace imaging {
  class grid_single_correlation {};
  class grid_duel_correlation {};
  class grid_4_correlation {};
  class grid_4_correlation_with_jones_corrections {};
  class grid_sampling_function{};
  template <typename correlation_gridding_mode>
  class correlation_gridding_traits {
    //Undefined base class
    class undefined_type {};
  public:
    typedef undefined_type vis_type;
    typedef undefined_type vis_flag_type;
    typedef undefined_type vis_weight_type;
    typedef undefined_type accumulator_type;
    typedef undefined_type normalization_accumulator_type;
  };
  template <>
  class correlation_gridding_traits<grid_single_correlation> {
  public:
    typedef vec1<basic_complex<visibility_base_type> > vis_type;
    typedef vec1<bool> vis_flag_type;
    typedef vec1<visibility_weights_base_type> vis_weight_type;
    typedef vec1<basic_complex<visibility_base_type> > accumulator_type;
    typedef vec1<normalization_base_type> normalization_accumulator_type;
  };
  template <>
  class correlation_gridding_traits<grid_duel_correlation> {
  public:
    typedef vec2<basic_complex<visibility_base_type> > vis_type;
    typedef vec2<bool> vis_flag_type;
    typedef vec2<visibility_weights_base_type> vis_weight_type;
    typedef vec2<basic_complex<visibility_base_type> > accumulator_type;
    typedef vec2<normalization_base_type> normalization_accumulator_type;
  };
  template <>
  class correlation_gridding_traits<grid_4_correlation> {
  public:
    typedef vec4<basic_complex<visibility_base_type> > vis_type;
    typedef vec4<bool> vis_flag_type;
    typedef vec4<visibility_weights_base_type> vis_weight_type;
    typedef vec4<basic_complex<visibility_base_type> > accumulator_type;
    typedef vec4<normalization_base_type> normalization_accumulator_type;
  };
  template <>
  class correlation_gridding_traits<grid_4_correlation_with_jones_corrections>:
	public correlation_gridding_traits<grid_4_correlation>{};
  template <>
  class correlation_gridding_traits<grid_sampling_function>:
	correlation_gridding_traits<grid_single_correlation>{}; //sampling function stays the same accross correlations
  /**
   * scalar multiplication with correlated visibilities (can be up to 4 complex visibilties)
   */
  template <typename T>
  __device__ __host__ vec1<basic_complex<T> > operator*(const vec1<basic_complex<T> > & visibilities, const vec1<T> & scalars) {
    return vec1<basic_complex<T> >(basic_complex<T>(visibilities._x._real*scalars._x,visibilities._x._imag*scalars._x));
  }
  template <typename T>
  __device__ __host__ vec2<basic_complex<T> > operator*(const vec2<basic_complex<T> > & visibilities, const vec2<T> & scalars) {
    return vec2<basic_complex<T> >(basic_complex<T>(visibilities._x._real*scalars._x,visibilities._x._imag*scalars._x),
				   basic_complex<T>(visibilities._y._real*scalars._y,visibilities._y._imag*scalars._y));
  }
  template <typename T>
  __device__ __host__ vec4<basic_complex<T> > operator*(const vec4<basic_complex<T> > & visibilities, const vec4<T> & scalars) {
    return vec4<basic_complex<T> >(basic_complex<T>(visibilities._x._real*scalars._x,visibilities._x._imag*scalars._x),
				   basic_complex<T>(visibilities._y._real*scalars._y,visibilities._y._imag*scalars._y),
				   basic_complex<T>(visibilities._z._real*scalars._z,visibilities._z._imag*scalars._z),
				   basic_complex<T>(visibilities._w._real*scalars._w,visibilities._w._imag*scalars._w));
  }
  /**
   * Define conjugates for the different correlations
   */
  template <typename T>
  __device__ __host__ void conj(vec1< basic_complex<T> > & visibilities){
    visibilities._x._imag *= -1;
  }
  template <typename T>
  __device__ __host__ void conj(vec2< basic_complex<T> > & visibilities){
    visibilities._x._imag *= -1;
    visibilities._y._imag *= -1;
  }
  template <typename T>
  __device__ __host__ void conj(vec4< basic_complex<T> > & visibilities){
    visibilities._x._imag *= -1;
    visibilities._y._imag *= -1;
    visibilities._z._imag *= -1;
    visibilities._w._imag *= -1;
  }
  /**
   * Multiply jones_2x2 matrix with vec4< basic_complex < T > >
   * Be careful to ensure commutivity: group your operators when doing a string of matrix multiplies!
   */
  template <typename T>
  __device__ __host__ vec4<basic_complex<T> > operator*(const jones_2x2<T> & jones, const vec4<basic_complex<T> > & vis){
    jones_2x2<T> rhs = *((jones_2x2<T>*)&vis); //structure is equivalent so just reinterpret cast
    jones_2x2<T> out;
    out.correlations[0] = jones.correlations[0]*rhs.correlations[0] + jones.correlations[1]*rhs.correlations[2];
    out.correlations[1] = jones.correlations[0]*rhs.correlations[1] + jones.correlations[1]*rhs.correlations[3];
    out.correlations[2] = jones.correlations[2]*rhs.correlations[0] + jones.correlations[3]*rhs.correlations[2];
    out.correlations[3] = jones.correlations[2]*rhs.correlations[1] + jones.correlations[3]*rhs.correlations[3];
    return *((vec4<basic_complex<T> >*)&out);
  }
  /**
   * Multiply jones_2x2 matrix with vec4< basic_complex < T > >
   * Be careful to ensure commutivity: group your operators when doing a string of matrix multiplies!
   */
  template <typename T>
  __device__ __host__ vec4<basic_complex<T> > operator*(const vec4<basic_complex<T> > & vis,const jones_2x2<T> & jones){
    jones_2x2<T> lhs = *((jones_2x2<T>*)&vis); //structure is equivalent so just reinterpret cast
    jones_2x2<T> out;
    out.correlations[0] = lhs.correlations[0]*jones.correlations[0] + lhs.correlations[1]*jones.correlations[2];
    out.correlations[1] = lhs.correlations[0]*jones.correlations[1] + lhs.correlations[1]*jones.correlations[3];
    out.correlations[2] = lhs.correlations[2]*jones.correlations[0] + lhs.correlations[3]*jones.correlations[2];
    out.correlations[3] = lhs.correlations[2]*jones.correlations[1] + lhs.correlations[3]*jones.correlations[3];
    return *((vec4<basic_complex<T> >*)&out);
  }
  
  __host__ inline void mul_vis_with_scalars(const vec1< basic_complex<float> > & vis_in, 
				     convolution_base_type conv_weight[4], 
				     vec1< basic_complex<visibility_base_type> > visses_out[4]){
    __m128 vis_2 = _mm_set_ps(vis_in._x._imag,vis_in._x._real,vis_in._x._imag,vis_in._x._real);
    _mm_store_ps((float*)(&visses_out[0]),
		 _mm_mul_ps(vis_2,
			    _mm_set_ps(conv_weight[1],conv_weight[1],conv_weight[0],conv_weight[0])));
    _mm_store_ps((float*)(&visses_out[2]),
		 _mm_mul_ps(vis_2,
			    _mm_set_ps(conv_weight[3],conv_weight[3],conv_weight[2],conv_weight[2])));
  }
};
