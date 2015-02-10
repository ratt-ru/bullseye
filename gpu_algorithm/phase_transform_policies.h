#pragma once
#include "gridding_parameters.h"
#include "cu_common.h"

namespace imaging {
  class disable_faceting_phase_shift {};
  class enable_faceting_phase_shift {};
  
  template <typename T>
  class phase_transform_policy{
  public:
    __device__ __host__ static void read_facet_ra_dec(const gridding_parameters & params, size_t facet_index, uvw_base_type & facet_ra, uvw_base_type & facet_dec);
  };
  template <>
  class phase_transform_policy<disable_faceting_phase_shift>{
  public:
    __device__ __host__ static void read_facet_ra_dec(const gridding_parameters & params, size_t facet_index, uvw_base_type & facet_ra, uvw_base_type & facet_dec){
      facet_ra = params.phase_centre_ra;
      facet_dec = params.phase_centre_dec;
    }
  };
  template <>
  class phase_transform_policy<enable_faceting_phase_shift>{
  public:
    __device__ __host__ static void read_facet_ra_dec(const gridding_parameters & params, size_t facet_index, uvw_base_type & facet_ra, uvw_base_type & facet_dec){
      size_t facet_centre_index = facet_index << 1;
      facet_ra = params.facet_centres[facet_centre_index];
      facet_dec = params.facet_centres[facet_centre_index + 1];
    }
  };
}