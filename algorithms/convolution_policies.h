#pragma once
#include <stdexcept>
#include <complex>
#include "uvw_coord.h"

namespace imaging {
  class convolution_precomputed_fir {};
  /**
   Reference convolution policy
   */
  template <typename convolution_base_type, typename uvw_base_type, typename grid_base_type,
	    typename gridding_policy_type, typename convolution_mode>
  class convolution_policy {
  public:
    convolution_policy() {
      throw std::runtime_error("Undefined behaviour");
    }
    /**
     Convolve will call a gridding function of the active gridding policy (this can be of the normal visibility or its conjungate)
     All gridding policies should therefore support gridding both visibility terms (see Synthesis Imaging II, pg. 25-26)
     
     All specializing policies must have a function conforming to the following function header:
     uvw_coord: reference to central uvw coord (in continious space)
     gridding_function: pointer to member function of active gridding policy
     */
    inline void convolve(const uvw_coord<uvw_base_type> & __restrict__ uvw,
			 const typename gridding_policy_type::trait_type::pol_vis_type & __restrict__ vis,
			 std::size_t no_grids_to_offset) const __restrict__ {
      throw std::runtime_error("Undefined behaviour");
    }
  };

  /**
   * Default oversampled convolution (using precomputed filter) of size "full support" * "oversampling factor" 
   */
  template <typename convolution_base_type, typename uvw_base_type, typename grid_base_type, typename gridding_policy_type>
  class convolution_policy <convolution_base_type, uvw_base_type, grid_base_type, gridding_policy_type, convolution_precomputed_fir> {
  private:
    std::size_t _nx;
    std::size_t _ny;
    std::size_t _grid_size_in_pixels;
    uvw_base_type _grid_u_centre;
    uvw_base_type _grid_v_centre;
    std::size_t _convolution_support;
    std::size_t _oversampling_factor;
    const convolution_base_type * __restrict__ _conv;
    std::size_t _conv_dim_size;
    uvw_base_type _conv_dim_centre;
    uvw_base_type _conv_scale;
    gridding_policy_type & __restrict__ _active_gridding_policy;
    std::size_t _cube_chan_dim_step;
  public:
    /**
     conv: precomputed convolution FIR of size (conv_support x conv_oversample)^2, flat-indexed
     conv_support, conv_oversample: integral numbers
     polarization_index: index of the polarization correlation term currently being gridded
     PRECONDITION
      1. (conv_support x conv_oversample)^2 == ||conv||
    */
    convolution_policy(std::size_t nx, std::size_t ny, std::size_t no_polarizations, std::size_t convolution_support, std::size_t oversampling_factor, 
		       const convolution_base_type * conv, gridding_policy_type & active_gridding_policy):
			_nx(nx), _ny(ny), _grid_size_in_pixels(nx*ny), _grid_u_centre(nx / 2.0), _grid_v_centre(ny / 2.0),
			_convolution_support(convolution_support), _oversampling_factor(oversampling_factor), 
			_conv(conv), _conv_dim_size(convolution_support * oversampling_factor),
			_conv_dim_centre(convolution_support / 2.0),
			_conv_scale(1/uvw_base_type(oversampling_factor)), //convolution pixel is oversample times smaller than scaled grid cell size			
			_active_gridding_policy(active_gridding_policy),
			_cube_chan_dim_step(nx*ny*no_polarizations)
			{}
    inline void convolve(const uvw_coord<uvw_base_type> & __restrict__ uvw,
			 const typename gridding_policy_type::trait_type::pol_vis_type & __restrict__ vis,
			 std::size_t no_grids_to_offset) const __restrict__ {
	uvw_base_type translated_grid_u = uvw._u + _grid_u_centre - _conv_dim_centre;
	uvw_base_type translated_grid_v = uvw._v + _grid_v_centre - _conv_dim_centre;
	std::size_t chan_offset = no_grids_to_offset * _cube_chan_dim_step;
	
        for (std::size_t conv_v = 0; conv_v < _conv_dim_size; ++conv_v) {
            std::size_t disc_grid_v = std::lrint(translated_grid_v + conv_v*_conv_scale);
            if (disc_grid_v >= _ny) continue;
	    std::size_t grid_flat_index_v = (disc_grid_v)*_nx;
	    std::size_t conv_flat_index_v = conv_v * _conv_dim_size;
	    for (int conv_u = 0; conv_u < _conv_dim_size; ++conv_u) {
                std::size_t disc_grid_u = std::lrint(translated_grid_u + conv_u*_conv_scale);
                if (disc_grid_u >= _nx) continue;
                //by definition the convolution FIR is 0 outside the support region:
                convolution_base_type conv_weight = _conv[conv_flat_index_v + conv_u];
                
                _active_gridding_policy.grid_polarization_terms(chan_offset + grid_flat_index_v + disc_grid_u, vis, conv_weight);
            }
        }
    }
  };
}