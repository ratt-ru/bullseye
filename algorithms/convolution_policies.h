#pragma once
#include <stdexcept>
#include "uvw_coord.h"

namespace imaging {
  class convolution_precomputed_fir {};
  /**
   Reference convolution policy
   */
  template <typename convolution_base_type, typename uvw_base_type, 
	    typename gridding_policy_type, typename convolution_mode>
  class convolution_policy {
  public:
    convolution_policy() {
      throw std::exception("Undefined behaviour");
    }
    /**
     Convolve will call a gridding function of the active gridding policy (this can be of the normal visibility or its conjungate)
     All gridding policies should therefore support gridding both visibility terms (see Synthesis Imaging II, pg. 25-26)
     
     All specializing policies must have a function conforming to the following function header:
     uvw_coord: reference to central uvw coord (in continious space)
     gridding_function: pointer to member function of active gridding policy
     */
    inline void convolve(const uvw_coord<uvw_base_type> & __restrict__ uvw,
			 void (gridding_policy_type::*gridding_function)(std::size_t,std::size_t,convolution_base_type)) const __restrict__ {
      throw std::exception("Undefined behaviour");
    }
  };
  
  /**
   * Default oversampled convolution (using precomputed filter) of size "full support" * "oversampling factor" 
   */
  template <typename convolution_base_type, typename uvw_base_type, typename gridding_policy_type>
  class convolution_policy <convolution_base_type, uvw_base_type, gridding_policy_type, convolution_precomputed_fir> {
  private:
    std::size_t _nx;
    std::size_t _ny;
    std::size_t _grid_size_in_pixels;
    std::size_t _grid_u_centre;
    std::size_t _grid_v_centre;
    std::size_t _convolution_support;
    std::size_t _oversampling_factor;
    const convolution_base_type * __restrict__ _conv;
    std::size_t _conv_dim_size;
    std::size_t _conv_dim_centre;
    uvw_base_type _conv_scale; 
    gridding_policy_type & __restrict__ _active_gridding_policy;
  public:
    /**
     conv: precomputed convolution FIR of size (conv_support x conv_oversample)^2, flat-indexed
     conv_support, conv_oversample: integral numbers
     polarization_index: index of the polarization correlation term currently being gridded
     PRECONDITION
      1. (conv_support x conv_oversample)^2 == ||conv||
    */
    convolution_policy(std::size_t nx, std::size_t ny, std::size_t convolution_support, std::size_t oversampling_factor, 
		       const convolution_base_type * conv, gridding_policy_type & active_gridding_policy):
			_nx(nx), _ny(ny), _grid_size_in_pixels(nx*ny), _grid_u_centre(nx / 2), _grid_v_centre(ny / 2),
			_convolution_support(convolution_support), _oversampling_factor(oversampling_factor), 
			_conv(conv), _conv_dim_size(convolution_support * oversampling_factor),
			_conv_dim_centre(_conv_dim_size / 2),
			_conv_scale(1/uvw_base_type(oversampling_factor)), //convolution pixel is oversample times smaller than scaled grid cell size
			_active_gridding_policy(active_gridding_policy)
			{}
    inline void convolve(const uvw_coord<uvw_base_type> & __restrict__ uvw,
			 void (gridding_policy_type::*gridding_function)(std::size_t,std::size_t,convolution_base_type)) const __restrict__ {
	/*
	 compute the distance the u,v coordinate is from the bin center, scaled by the oversampling factor
	 (size of the jump in imaging space). Then shift the uv coordinate to the grid centre
	*/
	uvw_base_type frac_u = _oversampling_factor*(uvw._u - (uvw_base_type)(int64_t)uvw._u); 
	uvw_base_type frac_v = _oversampling_factor*(uvw._v - (uvw_base_type)(int64_t)uvw._v); 
	uvw_base_type translated_grid_u = uvw._u + _grid_u_centre;
	uvw_base_type translated_grid_v = uvw._v + _grid_v_centre;
	
        for (std::size_t conv_v = 0; conv_v < _conv_dim_size; ++conv_v) {
            std::size_t disc_grid_v = translated_grid_v + (conv_v - _conv_dim_centre)*_conv_scale;
            if (disc_grid_v >= _ny) continue;

            std::size_t offset_conv_v = frac_v + conv_v;
            for (std::size_t conv_u = 0; conv_u < _conv_dim_size; ++conv_u) {
                std::size_t disc_grid_u = translated_grid_u + (conv_u - _conv_dim_centre)*_conv_scale;
                if (disc_grid_u >= _nx) continue;

                std::size_t offset_conv_u = frac_u + conv_u;
                std::size_t conv_flat_index = ((offset_conv_v) * _conv_dim_size + offset_conv_u); //flatten convolution index
                //by definition the convolution FIR is 0 outside the support region:
                convolution_base_type conv_weight = (convolution_base_type) ( (offset_conv_u < _conv_dim_size &&
                                                    offset_conv_v < _conv_dim_size) ? _conv[conv_flat_index] : 0);
                std::size_t grid_flat_index = ((disc_grid_v)*_nx+(disc_grid_u)); //flatten grid index
                
                // Call the gridding policy function (this can either be the normal gridding function or the conjugate gridding function:
                ((_active_gridding_policy).*gridding_function)(grid_flat_index,_grid_size_in_pixels,conv_weight);
            }
        }
    }
  };
}