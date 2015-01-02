#pragma once
#include <stdexcept>
#include <complex>
#include "uvw_coord.h"
#include "fft_shift_utils.h"

#include <boost/math/special_functions/bessel.hpp>
#include <boost/math/special_functions/factorials.hpp>
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
    uvw_base_type _conv_centre_offset;
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
			_convolution_support(convolution_support*2 + 1), _oversampling_factor(oversampling_factor), 
			_conv(conv), _conv_dim_size(_convolution_support * _oversampling_factor),
			_conv_centre_offset((convolution_support)/2.0),
			_active_gridding_policy(active_gridding_policy),
			_cube_chan_dim_step(nx*ny*no_polarizations)
			{}
    inline void convolve(const uvw_coord<uvw_base_type> & __restrict__ uvw,
			 const typename gridding_policy_type::trait_type::pol_vis_type & __restrict__ vis,
			 std::size_t no_grids_to_offset) const __restrict__ {
	auto convolve = [this](convolution_base_type x){
// 	  convolution_base_type beta = 4.686466667f * this->_convolution_support - 0.3422f; //regression line
// 	  convolution_base_type sqr_term = (2 * x / this->_convolution_support);
// 	  convolution_base_type sqrt_term = 1 - sqr_term * sqr_term;
	  /*funciton is only defined within -support*0.5 <= x <= support*0.5
	   *introduce a boxcar to ensure we dont fall off that support region
	   *this means however that we're tapering with a sinc function in the intensity domain
	   */ 
// 	  if (sqrt_term < 0)
// 	    return (convolution_base_type)0.0;
// 	  return (convolution_base_type)boost::math::cyl_bessel_j<convolution_base_type>(0,beta * sqrt(sqrt_term))/this->_convolution_support;
	  //This crap introduces a stuff load of radial tapering in the intensity domain... only the kaiser bessel really work as advertised:
// 	  #define ALPHA 0.4463
// 	  return ALPHA + (1 - ALPHA) * cos(2*M_PI*x/this->_convolution_support);
	  convolution_base_type sigma =  0.0349*x + 0.37175;
	  return exp(-0.5 * (x/sigma)*(x/sigma));
	};
	std::size_t chan_offset = no_grids_to_offset * _cube_chan_dim_step;

	uvw_base_type translated_grid_u = uvw._u + _grid_u_centre - _conv_centre_offset;
	uvw_base_type translated_grid_v = uvw._v + _grid_v_centre - _conv_centre_offset;
	std::size_t disc_grid_u = std::lrint(translated_grid_u);
	std::size_t disc_grid_v = std::lrint(translated_grid_v);
	
	if (disc_grid_v + _convolution_support  >= _ny || disc_grid_u + _convolution_support  >= _nx) return;
	{
	    uvw_base_type frac_u = (-translated_grid_u + (uvw_base_type)disc_grid_u);
	    uvw_base_type frac_v = (-translated_grid_v + (uvw_base_type)disc_grid_v);
	    
            for (std::size_t sup_v = 0; sup_v < _convolution_support; ++sup_v) {
                std::size_t convolved_grid_v = disc_grid_v + sup_v;
                uvw_base_type conv_v = (uvw_base_type)sup_v - _conv_centre_offset + frac_v;
                for (int sup_u = 0; sup_u < _convolution_support; ++sup_u) {
                    std::size_t convolved_grid_u = disc_grid_u + sup_u;
		    uvw_base_type conv_u = (uvw_base_type)sup_u - _conv_centre_offset + frac_u;
                    std::size_t grid_flat_index = convolved_grid_v*_ny + convolved_grid_u;

                    convolution_base_type conv_weight = convolve(conv_v) * convolve(conv_u);
                    _active_gridding_policy.grid_polarization_terms(chan_offset + grid_flat_index, vis, conv_weight);
                }
            }
	}
// 	{
// 	    uvw_base_type frac_u = (translated_grid_u - (uvw_base_type)disc_grid_u) + 1;
// 	    uvw_base_type frac_v = (translated_grid_v - (uvw_base_type)disc_grid_v) + 1;
// 	    
//             for (std::size_t sup_v = 0; sup_v <= _convolution_support/2; ++sup_v) {
//                 std::size_t convolved_grid_v = disc_grid_v + sup_v + 1;
//                 uvw_base_type conv_v = (uvw_base_type)sup_v - _conv_centre_offset + frac_v;
//                 for (int sup_u = 0; sup_u <= _convolution_support/2; ++sup_u) {
//                     std::size_t convolved_grid_u = disc_grid_u + sup_u + 1;
// 		    uvw_base_type conv_u = (uvw_base_type)sup_u - _conv_centre_offset + frac_u;
//                     std::size_t grid_flat_index = convolved_grid_v*_ny + convolved_grid_u;
// 
//                     convolution_base_type conv_weight = convolve(conv_v) * convolve(conv_u);
//                     _active_gridding_policy.grid_polarization_terms(chan_offset + grid_flat_index, vis, conv_weight);
//                 }
//             }
// 	}
    }
  };
}