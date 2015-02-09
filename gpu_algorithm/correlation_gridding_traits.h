#pragma once
#include "cu_common.h"
#include "gpu_wrapper.h"
#include "cu_vec.h"
#include "cu_basic_complex.h"
namespace imaging {
  class grid_single_correlation {};
  class grid_duel_correlation {};
  class grid_4_correlation {};
  
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
    typedef vec1<vec2<visibility_base_type> > vis_type;
    typedef vec1<bool> vis_flag_type;
    typedef vec1<visibility_weights_base_type> vis_weight_type;
    typedef vec1<vec2<visibility_base_type> > accumulator_type;
  };
  template <>
  class correlation_gridding_traits<grid_duel_correlation> {
  public:
    typedef vec2<vec2<visibility_base_type> > vis_type;
    typedef vec2<bool> vis_flag_type;
    typedef vec2<visibility_weights_base_type> vis_weight_type;
    typedef vec2<vec2<visibility_base_type> > accumulator_type;
  };
  template <>
  class correlation_gridding_traits<grid_4_correlation> {
  public:
    typedef vec4<vec2<visibility_base_type> > vis_type;
    typedef vec4<bool> vis_flag_type;
    typedef vec4<visibility_weights_base_type> vis_weight_type;
    typedef vec4<vec2<visibility_base_type> > accumulator_type;
  };
  
  /**
   * scalar multiplication with correlated visibilities (can be up to 4 complex visibilties)
   */
  template <typename T>
  __device__ __host__ vec1<vec2<T> > operator*(const vec1<vec2<T> > & visibilities, const vec1<T> & scalars) {
    return vec1<vec2<T> >(vec2<T>(visibilities._x._x*scalars._x,visibilities._x._y*scalars._x));
  }
  template <typename T>
  __device__ __host__ vec2<vec2<T> > operator*(const vec2<vec2<T> > & visibilities, const vec2<T> & scalars) {
    return vec2<vec2<T> >(vec2<T>(visibilities._x._x*scalars._x,visibilities._x._y*scalars._x),
			vec2<T>(visibilities._y._x*scalars._y,visibilities._y._y*scalars._y));
  }
  template <typename T>
  __device__ __host__ vec4<vec2<T> > operator*(const vec4<vec2<T> > & visibilities, const vec4<T> & scalars) {
    return vec4<vec2<T> >(vec2<T>(visibilities._x._x*scalars._x,visibilities._x._y*scalars._x),
			vec2<T>(visibilities._y._x*scalars._y,visibilities._y._y*scalars._y),
			vec2<T>(visibilities._z._x*scalars._z,visibilities._z._y*scalars._z),
			vec2<T>(visibilities._w._x*scalars._w,visibilities._w._y*scalars._w));
  }
};
