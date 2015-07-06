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
#include "gridding_parameters.h"
#include "cu_common.h"
#include "base_types.h"
#include "uvw_coord.h"
#include "cu_basic_complex.h"
namespace imaging {
  extern surface<void, cudaSurfaceType2D> cached_convolution_functions;
  
  class AA_1D_precomputed {};
  class W_projection_1D_precomputed {};
  
  template <typename active_correlation_gridding_policy, typename policy_type>
  class convolution_policy {
  public:
    __device__ static void compute_closest_uv_in_conv_kernel(const gridding_parameters & params,
							     typename active_correlation_gridding_policy::active_trait::vis_type & out_vis,
							     imaging::uvw_coord<uvw_base_type> & out_uvw,
							     uvw_base_type grid_centre_offset_x,uvw_base_type grid_centre_offset_y,
							     size_t my_conv_u,size_t my_conv_v,
							     size_t & out_my_current_u, size_t & out_my_current_v,
							     size_t & out_closest_conv_u, size_t & out_closest_conv_v);
    __device__ static void convolve(
			       const gridding_parameters & params,
			       const imaging::uvw_coord<uvw_base_type> & uvw,
			       const typename active_correlation_gridding_policy::active_trait::vis_type & vis,
			       typename active_correlation_gridding_policy::active_trait::accumulator_type & my_grid_accum,//out
			       typename active_correlation_gridding_policy::active_trait::normalization_accumulator_type & normalization_term,//out
			       const typename active_correlation_gridding_policy::active_trait::vis_weight_type & weight,
			       size_t closest_conv_u, //best point on the oversampled convolution filter
			       size_t closest_conv_v); //best point on the oversampled convolution filter
  };
  /**
   * This is a policy to deal with 1D precomputed Anti-Aliasing (real-valued) filters
   */
  template <typename active_correlation_gridding_policy>
  class convolution_policy <active_correlation_gridding_policy,AA_1D_precomputed> {
  public:
    __device__ static void compute_closest_uv_in_conv_kernel(const gridding_parameters & params,
							     typename active_correlation_gridding_policy::active_trait::vis_type & out_vis,
							     imaging::uvw_coord<uvw_base_type> & out_uvw,
							     uvw_base_type grid_centre_offset_x,uvw_base_type grid_centre_offset_y,
							     size_t my_conv_u,size_t my_conv_v,
							     size_t & out_my_current_u, size_t & out_my_current_v,
							     size_t & out_closest_conv_u, size_t & out_closest_conv_v)
    {
      uvw_base_type cont_current_u = out_uvw._u + grid_centre_offset_x;
      uvw_base_type cont_current_v = out_uvw._v + grid_centre_offset_y;
      out_my_current_u = (signbit(cont_current_u) == 0) ? round(cont_current_u) : params.nx; //underflow of size_t seems to be always == 0 in nvcc
      out_my_current_v = (signbit(cont_current_v) == 0) ? round(cont_current_v) : params.ny; //underflow of size_t seems to be always == 0 in nvcc
      size_t frac_u_offset = (1-out_uvw._u + round(out_uvw._u)) * params.conv_oversample; //1 + frac_u in filter samples
      size_t frac_v_offset = (1-out_uvw._v + round(out_uvw._v)) * params.conv_oversample; //1 + frac_u in filter samples
      out_closest_conv_u = frac_u_offset + my_conv_u * params.conv_oversample;
      out_closest_conv_v = frac_v_offset + my_conv_v * params.conv_oversample;
    }
    __device__ static void convolve(
			       const gridding_parameters & params,
			       const imaging::uvw_coord<uvw_base_type> & uvw,
			       const typename active_correlation_gridding_policy::active_trait::vis_type & vis,
			       typename active_correlation_gridding_policy::active_trait::accumulator_type & my_grid_accum,//out
			       typename active_correlation_gridding_policy::active_trait::normalization_accumulator_type & normalization_term,//out
			       const typename active_correlation_gridding_policy::active_trait::vis_weight_type & weight,
			       size_t closest_conv_u, //best point on the oversampled convolution filter
			       size_t closest_conv_v) //best point on the oversampled convolution filter
    {
      convolution_base_type conv_u;
      convolution_base_type conv_v;
      surf2Dread<convolution_base_type>(&conv_u,cached_convolution_functions,closest_conv_u * sizeof(convolution_base_type),0,cudaBoundaryModeTrap);
      surf2Dread<convolution_base_type>(&conv_v,cached_convolution_functions,closest_conv_v * sizeof(convolution_base_type),0,cudaBoundaryModeTrap);
      convolution_base_type conv_weight = conv_u * conv_v;
      typename active_correlation_gridding_policy::active_trait::vis_weight_type convolve_weight = weight * conv_weight; //compute the weighted convolution weight from seperable 1D filter
      //then multiply-add into the accumulator
      my_grid_accum += vis * convolve_weight;
      normalization_term += vector_promotion<visibility_weights_base_type,normalization_base_type>(convolve_weight);
    }
  };
  
  /**
   * This is a policy to deal with seperable w-projection (complex-valued) filters
   * Note: these filters are not as accurate as their non-seperable counterparts, but
   * for faceting purposes (and for telescopes with reasonably narrowish fields of view)
   * they should be good enough. It is critical to use the seperable filters, because we
   * can't fit them into texture/surface memory. Using uncached gridding filters slows
   * down gridding considerably
   */
  template <typename active_correlation_gridding_policy>
  class convolution_policy <active_correlation_gridding_policy,W_projection_1D_precomputed> {
  public:
    __device__ static void compute_closest_uv_in_conv_kernel(const gridding_parameters & params,
							     typename active_correlation_gridding_policy::active_trait::vis_type & out_vis,
							     imaging::uvw_coord<uvw_base_type> & out_uvw,
							     uvw_base_type grid_centre_offset_x,uvw_base_type grid_centre_offset_y,
							     size_t my_conv_u,size_t my_conv_v,
							     size_t & out_my_current_u, size_t & out_my_current_v,
							     size_t & out_closest_conv_u, size_t & out_closest_conv_v)
    {
      //When doing w-projection W should be positive (either we grid the visibility or its conjugate baseline)
      //in order to save on computing twice the number of w layers!
      if (out_uvw._w < 0){
	conj<visibility_base_type>(out_vis);
	out_uvw._u *= -1;
	out_uvw._v *= -1;
	out_uvw._w *= -1;
      }
      //now compute the interpolation error as per normal gridding:
      convolution_policy< active_correlation_gridding_policy, AA_1D_precomputed >::compute_closest_uv_in_conv_kernel(params,
														     out_vis,
														     out_uvw,
														     grid_centre_offset_x,
														     grid_centre_offset_y,
														     my_conv_u,my_conv_v,
														     out_my_current_u,
														     out_my_current_v,
														     out_closest_conv_u,
														     out_closest_conv_v);
    }
    __device__ static void convolve(
			       const gridding_parameters & params,
			       const imaging::uvw_coord<uvw_base_type> & uvw,
			       const typename active_correlation_gridding_policy::active_trait::vis_type & vis,
			       typename active_correlation_gridding_policy::active_trait::accumulator_type & my_grid_accum,//out
			       typename active_correlation_gridding_policy::active_trait::normalization_accumulator_type & normalization_term,//out
			       const typename active_correlation_gridding_policy::active_trait::vis_weight_type & weight,
			       size_t closest_conv_u, //best point on the oversampled convolution filter
			       size_t closest_conv_v) //best point on the oversampled convolution filter
    {
      basic_complex<convolution_base_type> conv_u;
      basic_complex<convolution_base_type> conv_v;
      size_t best_fit_w_plane = round(abs(uvw._w)/(float)params.wmax_est*(params.wplanes-1));
      surf2Dread<basic_complex<convolution_base_type> >(&conv_u,cached_convolution_functions,
							closest_conv_u * sizeof(basic_complex<convolution_base_type>),best_fit_w_plane,
							cudaBoundaryModeTrap);
      surf2Dread<basic_complex<convolution_base_type> >(&conv_v,cached_convolution_functions,
							closest_conv_v * sizeof(basic_complex<convolution_base_type>),best_fit_w_plane,
							cudaBoundaryModeTrap);
      basic_complex<convolution_base_type> conv_weight = conv_u * conv_v;
      //then multiply-add into the accumulator
      my_grid_accum += vis * conv_weight * weight;
      normalization_term += vector_promotion<visibility_weights_base_type,normalization_base_type>(weight)*conv_weight._real; //convolution real and imaginary parts are similar
    }
  };
}