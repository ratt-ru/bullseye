#pragma once
#include "cu_common.h"
#include "gpu_wrapper.h"
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
  };
  template <>
  class correlation_gridding_traits<grid_single_correlation> {
  public:
    typedef vec1<basic_complex<visibility_base_type> > vis_type;
    typedef vec1<bool> vis_flag_type;
    typedef vec1<visibility_weights_base_type> vis_weight_type;
    typedef vec1<basic_complex<visibility_base_type> > accumulator_type;
  };
  template <>
  class correlation_gridding_traits<grid_duel_correlation> {
  public:
    typedef vec2<basic_complex<visibility_base_type> > vis_type;
    typedef vec2<bool> vis_flag_type;
    typedef vec2<visibility_weights_base_type> vis_weight_type;
    typedef vec2<basic_complex<visibility_base_type> > accumulator_type;
  };
  template <>
  class correlation_gridding_traits<grid_4_correlation> {
  public:
    typedef vec4<basic_complex<visibility_base_type> > vis_type;
    typedef vec4<bool> vis_flag_type;
    typedef vec4<visibility_weights_base_type> vis_weight_type;
    typedef vec4<basic_complex<visibility_base_type> > accumulator_type;
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
};
