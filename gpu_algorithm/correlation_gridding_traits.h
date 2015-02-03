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
    template<typename T> struct vec4 {T corr_1; T corr_2; T corr_3; T corr_4;};
  public:
    typedef vec4<vec2<visibility_base_type> > vis_type;
    typedef vec4<bool> vis_flag_type;
    typedef vec4<visibility_weights_base_type> vis_weight_type;
    typedef vec4<vec2<visibility_base_type> > accumulator_type;
  };
};