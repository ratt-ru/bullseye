#pragma once
#include <complex>
#include "jones_2x2.h"
namespace imaging {
  class gridding_single_pol {};
  class gridding_double_pol {};
  class gridding_4_pol {};
  class gridding_4_pol_enable_facet_based_jones_corrections {};
  class gridding_sampling_function {};
  template <typename visibility_base_type,typename weights_base_type,typename T>
  class polarization_gridding_trait {
    //Undefined base class
    class undefined_type {};
  public:
    typedef undefined_type pol_vis_type;
    typedef undefined_type pol_vis_weight_type;
    typedef undefined_type pol_vis_flag_type;
  };
  
  template <typename visibility_base_type,typename weights_base_type>
  class polarization_gridding_trait<visibility_base_type,weights_base_type,gridding_single_pol> {
  public:
    typedef std::complex<visibility_base_type> pol_vis_type;
    typedef weights_base_type pol_vis_weight_type;
    typedef bool pol_vis_flag_type;
  };
  
  template <typename visibility_base_type,typename weights_base_type>
  class polarization_gridding_trait<visibility_base_type,weights_base_type,gridding_sampling_function>{
    public:
      typedef visibility_base_type pol_vis_type;
      typedef weights_base_type pol_vis_weight_type;
      typedef bool pol_vis_flag_type;
  };
  
  template <typename visibility_base_type,typename weights_base_type>
  class polarization_gridding_trait<visibility_base_type,weights_base_type,gridding_double_pol> {
  public:
    typedef struct pol_vis_type {std::complex<visibility_base_type> v[2]; } pol_vis_type;
    typedef struct pol_vis_weight_type { weights_base_type w[2]; } pol_vis_weight_type;
    typedef struct pol_vis_flag_type { bool f[2]; } pol_vis_flag_type;
  };
  
  template <typename visibility_base_type,typename weights_base_type>
  class polarization_gridding_trait<visibility_base_type,weights_base_type,gridding_4_pol> {
  public:
    typedef jones_2x2<visibility_base_type> pol_vis_type;
    typedef struct pol_vis_weight_type { weights_base_type w[4]; } pol_vis_weight_type;
    typedef struct pol_vis_flag_type { bool f[4]; } pol_vis_flag_type;
  };
  
  template <typename visibility_base_type,typename weights_base_type>
  class polarization_gridding_trait<visibility_base_type,weights_base_type,gridding_4_pol_enable_facet_based_jones_corrections> : 
    public polarization_gridding_trait<visibility_base_type,weights_base_type,gridding_4_pol> {
  };
}