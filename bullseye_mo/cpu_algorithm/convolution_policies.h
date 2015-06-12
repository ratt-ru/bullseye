#pragma once
#include "uvw_coord.h"
#include "gridding_parameters.h"
#include <cmath>
#include "cu_basic_complex.h"

namespace imaging {
class convolution_analytic_AA {};
class convolution_AA_1D_precomputed {};
class convolution_w_projection_precomputed {};
class convolution_NN {};
template <typename active_correlation_gridding_policy,typename T>
class convolution_policy {
public:
    inline static void set_required_rounding_operation();
    inline static void convolve(gridding_parameters & params, uvw_base_type grid_centre_offset_x,
                                uvw_base_type grid_centre_offset_y,
                                grid_base_type * __restrict__ facet_output_buffer,
				std::size_t channel_grid_index,
                                std::size_t grid_size_in_floats,
				size_t conv_full_support,
				size_t padded_conv_full_support,
				uvw_coord< uvw_base_type > & uvw,
                                typename active_correlation_gridding_policy::active_trait::vis_type & vis,
                                typename active_correlation_gridding_policy::active_trait::normalization_accumulator_type & normalization_term);
};

template <typename active_correlation_gridding_policy>
class convolution_policy <active_correlation_gridding_policy,convolution_NN>{
public:
    inline static void set_required_rounding_operation(){
      std::fesetround(FE_TONEAREST);
    }
    inline static void convolve(gridding_parameters & params, uvw_base_type grid_centre_offset_x,
                                uvw_base_type grid_centre_offset_y,
                                grid_base_type * __restrict__ facet_output_buffer,
				std::size_t channel_grid_index,
                                std::size_t grid_size_in_floats,
				size_t conv_full_support,
				size_t padded_conv_full_support,
				uvw_coord< uvw_base_type > & uvw,
                                typename active_correlation_gridding_policy::active_trait::vis_type & vis,
                                typename active_correlation_gridding_policy::active_trait::normalization_accumulator_type & normalization_term){
	uvw_base_type translated_grid_u = uvw._u + (params.nx >> 1);
        uvw_base_type translated_grid_v = uvw._v + (params.ny >> 1);
        std::size_t  disc_grid_u = std::lrint(translated_grid_u);
        std::size_t  disc_grid_v = std::lrint(translated_grid_v);
	//Don't you dare go over the boundary
        if (disc_grid_v >= params.ny || disc_grid_u >= params.nx) return; //negatives will be very big numbers... this is uints
	active_correlation_gridding_policy::grid_visibility(facet_output_buffer,
                        grid_size_in_floats,
                        params.nx,
                        channel_grid_index,
                        params.number_of_polarization_terms_being_gridded,
                        disc_grid_u,
                        disc_grid_v,
                        vis);
	normalization_term += 1.0;
    }
};
/**
 * This is simple precomputed convolution logic. This assumes a 1D (seperable 2D) real-valued
 * filter (such as a simple Anti-Aliasing filter)
 */
template <typename active_correlation_gridding_policy>
class convolution_policy <active_correlation_gridding_policy,convolution_AA_1D_precomputed> {
public:
    inline static void set_required_rounding_operation(){
      std::fesetround(FE_DOWNWARD); // this is the same strategy followed in the casacore gridder and produces very similar looking images
    }
    inline static void convolve(gridding_parameters & params, uvw_base_type grid_centre_offset_x,
                                uvw_base_type grid_centre_offset_y,
                                grid_base_type * __restrict__ facet_output_buffer,
				std::size_t channel_grid_index,
                                std::size_t grid_size_in_floats,
				size_t conv_full_support,
				size_t padded_conv_full_support,
				uvw_coord< uvw_base_type > & uvw,
                                typename active_correlation_gridding_policy::active_trait::vis_type & vis,
                                typename active_correlation_gridding_policy::active_trait::normalization_accumulator_type & normalization_term) {
        //account for interpolation error (we select the closest sample from the oversampled convolution filter)
        uvw_base_type translated_grid_u = uvw._u + grid_centre_offset_x;
        uvw_base_type translated_grid_v = uvw._v + grid_centre_offset_y;
        std::size_t  disc_grid_u = std::lrint(translated_grid_u);
        std::size_t  disc_grid_v = std::lrint(translated_grid_v);
        //to reduce the interpolation error we need to take the offset from the grid centre into account when choosing a convolution weight
        uvw_base_type frac_u = -translated_grid_u + (uvw_base_type)disc_grid_u;
        uvw_base_type frac_v = -translated_grid_v + (uvw_base_type)disc_grid_v;
        //Don't you dare go over the boundary
        if (disc_grid_v + padded_conv_full_support  >= params.ny || disc_grid_u + padded_conv_full_support >= params.nx ||
                disc_grid_v >= params.ny || disc_grid_u >= params.nx) return;

        std::size_t conv_v = (frac_v + 1) * params.conv_oversample;
        std::size_t  convolved_grid_v = (disc_grid_v + 1)*params.nx;
        for (std::size_t  sup_v = 0; sup_v < conv_full_support; ++sup_v) { //remember we have a +/- frac at both ends of the filter
            convolution_base_type conv_v_weight = params.conv[conv_v];
            std::size_t conv_u = (frac_u + 1) * params.conv_oversample;
            for (std::size_t sup_u = 0; sup_u < conv_full_support; ++sup_u) { //remember we have a +/- frac at both ends of the filter
                std::size_t convolved_grid_u = disc_grid_u + sup_u;
                convolution_base_type conv_u_weight = params.conv[conv_u];
                std::size_t grid_flat_index = convolved_grid_v + convolved_grid_u;

                convolution_base_type conv_weight = conv_u_weight * conv_v_weight;
                typename active_correlation_gridding_policy::active_trait::vis_type convolved_vis = vis * conv_weight;
                active_correlation_gridding_policy::grid_visibility(facet_output_buffer,
                        grid_size_in_floats,
                        params.nx,
                        channel_grid_index,
                        params.number_of_polarization_terms_being_gridded,
                        disc_grid_u + sup_u,
                        disc_grid_v + sup_v,
                        convolved_vis);
                normalization_term += conv_weight;
                conv_u += params.conv_oversample;
            } //conv_u
            conv_v += params.conv_oversample;
            convolved_grid_v += params.nx;
        } //conv_v
    }
};
/**
 * This is a simple 2D w-projection kernel
 */
template <typename active_correlation_gridding_policy>
class convolution_policy <active_correlation_gridding_policy,convolution_w_projection_precomputed> {
public:
    inline static void set_required_rounding_operation(){
      std::fesetround(FE_TONEAREST); // this is the same strategy followed in the casacore gridder and produces very similar looking images
    }
    inline static void convolve(gridding_parameters & params, uvw_base_type grid_centre_offset_x,
                                uvw_base_type grid_centre_offset_y,
                                grid_base_type * __restrict__ facet_output_buffer,
				std::size_t channel_grid_index,
                                std::size_t grid_size_in_floats,
				size_t conv_full_support,
				size_t padded_conv_full_support,
				uvw_coord< uvw_base_type > & uvw,
                                typename active_correlation_gridding_policy::active_trait::vis_type & vis,
                                typename active_correlation_gridding_policy::active_trait::normalization_accumulator_type & normalization_term) {
        //W should be positive (either we grid the visibility or it's conjugate baseline):	
	if (uvw._w < 0){
	  conj<visibility_base_type>(vis);  
	}
	//account for interpolation error (we select the closest sample from the oversampled convolution filter)
        uvw_base_type translated_grid_u = uvw._u + grid_centre_offset_x;
        uvw_base_type translated_grid_v = uvw._v + grid_centre_offset_y;
        std::size_t  disc_grid_u = std::lrint(translated_grid_u);
        std::size_t  disc_grid_v = std::lrint(translated_grid_v);
        //to reduce the interpolation error we need to take the offset from the grid centre into account when choosing a convolution weight
        uvw_base_type frac_u = -translated_grid_u + (uvw_base_type)disc_grid_u;
        uvw_base_type frac_v = -translated_grid_v + (uvw_base_type)disc_grid_v;
        
	std::size_t conv_dim_size = padded_conv_full_support + (padded_conv_full_support - 1) * (params.conv_oversample - 1);
	std::size_t best_fit_w_plane = std::lrint(fabs(uvw._w)/(float)params.wmax_est*(params.wplanes-1));
	std::size_t filter_offset = best_fit_w_plane * conv_dim_size * conv_dim_size;
	
	//Don't you dare go over the boundary
        if (disc_grid_v + padded_conv_full_support >= params.ny || disc_grid_u + padded_conv_full_support >= params.nx ||
                disc_grid_v >= params.ny || disc_grid_u >= params.nx || best_fit_w_plane >= params.wplanes) return;
	for (std::size_t sup_v = 0; sup_v < conv_full_support; ++sup_v){
	  std::size_t conv_v = std::size_t((uvw_base_type)(sup_v + 1) * params.conv_oversample + std::lrint(frac_v * params.conv_oversample));
	  for (std::size_t sup_u = 0; sup_u < conv_full_support; ++sup_u){
	      std::size_t conv_u = std::size_t((uvw_base_type)(sup_u + 1) * params.conv_oversample + std::lrint(frac_u * params.conv_oversample));
	      std::size_t conv_flat_index = filter_offset + conv_v * conv_dim_size + conv_u;
	      basic_complex<convolution_base_type> conv_weight = ((basic_complex<convolution_base_type>*)params.conv)[conv_flat_index];
	      typename active_correlation_gridding_policy::active_trait::vis_type convolved_vis = vis * conv_weight;
	      active_correlation_gridding_policy::grid_visibility(facet_output_buffer,
								  grid_size_in_floats,
								  params.nx,
								  channel_grid_index,
								  params.number_of_polarization_terms_being_gridded,
								  disc_grid_u + sup_u,
								  disc_grid_v + sup_v,
								  convolved_vis);
	      normalization_term += 1;//sqrt(conv_weight._real * conv_weight._real);
	  }
	}
// 	{
// 	//setup the iterators
// 	std::size_t conv_dim_size = padded_conv_full_support * params.conv_oversample;
// 	std::size_t conv_v = (frac_v + 1) * params.conv_oversample;
// 	std::size_t conv_v_flat_index = conv_v * conv_dim_size;
// 	//W should already be positive (either we grid the visibility or it's conjugate baseline):
// 	uvw_base_type clamped_w = fmin(uvw._w,params.wmax_est);
// 	std::size_t best_fit_w_plane = clamped_w / params.wmax_est * (params.wplanes - 1);
// 	std::size_t filter_offset = best_fit_w_plane * conv_dim_size * conv_dim_size;
// 	conv_v_flat_index += filter_offset; //lookup correct w filter
//         for (std::size_t  sup_v = 1; sup_v <= conv_full_support; ++sup_v) { //remember we have a +/- frac at both ends of the filter
//             std::size_t conv_u = (frac_u + 1) * params.conv_oversample;
//             for (std::size_t sup_u = 1; sup_u <= conv_full_support; ++sup_u) { //remember we have a +/- frac at both ends of the filter
//                 basic_complex<convolution_base_type> conv_weight = ((basic_complex<convolution_base_type>*)params.conv)[conv_v_flat_index + conv_u];
//                 typename active_correlation_gridding_policy::active_trait::vis_type convolved_vis = vis * conv_weight;
//                 active_correlation_gridding_policy::grid_visibility(facet_output_buffer,
//                         grid_size_in_floats,
//                         params.nx,
//                         channel_grid_index,
//                         params.number_of_polarization_terms_being_gridded,
//                         disc_grid_u + sup_u,
//                         disc_grid_v + sup_v,
//                         convolved_vis);
//                 //normalization_term += conv_weight;
//                 conv_u += params.conv_oversample;
//             } //conv_u
//             conv_v += params.conv_oversample;
// 	    conv_v_flat_index = conv_v * conv_dim_size;
//         } //conv_v
//         normalization_term += 1.0;
// 	}
    }
};
/**
 * This is a simple analytic convolution policy
 * Note that this assumes that FE_TONEAREST is set as the default rounding operation - there is a noticable phase error if not!!!
 */
template <typename active_correlation_gridding_policy>
class convolution_policy <active_correlation_gridding_policy,convolution_analytic_AA> {
public:
    inline static void set_required_rounding_operation(){
      std::fesetround(FE_TONEAREST);
    }
    inline static void convolve(gridding_parameters & params, uvw_base_type grid_centre_offset_x,
                                uvw_base_type grid_centre_offset_y,
                                grid_base_type * __restrict__ facet_output_buffer,
				std::size_t channel_grid_index,
                                std::size_t grid_size_in_floats,
				size_t conv_full_support,
				size_t padded_conv_full_support,
				uvw_coord< uvw_base_type > & uvw,
                                typename active_correlation_gridding_policy::active_trait::vis_type & vis,
                                typename active_correlation_gridding_policy::active_trait::normalization_accumulator_type & normalization_term) {
	auto convolve = [](convolution_base_type x)->convolution_base_type{
	  //sinc works okay
	  if (x != 0){
	    convolution_base_type param = M_PI*(x);
	    return (convolution_base_type)sin(param) / param;
	  } else {
	    return (convolution_base_type)1; //remove discontinuity
	  }
	};
	uvw_base_type translated_grid_u = uvw._u + params.nx/2 - (uvw_base_type)params.conv_support;
	uvw_base_type translated_grid_v = uvw._v + params.ny/2 - (uvw_base_type)params.conv_support;
	std::size_t disc_grid_u = round(translated_grid_u);
	std::size_t disc_grid_v = round(translated_grid_v);
	uvw_base_type frac_u = -translated_grid_u + (uvw_base_type)disc_grid_u;
	uvw_base_type frac_v = -translated_grid_v + (uvw_base_type)disc_grid_v;
	if (disc_grid_v + conv_full_support >= params.ny || disc_grid_u + conv_full_support >= params.nx ||
	  disc_grid_v >= params.ny || disc_grid_u >= params.nx) return;
	{
            for (std::size_t sup_v = 0; sup_v < conv_full_support; ++sup_v) {
		uvw_base_type conv_v = convolve(sup_v - (uvw_base_type)params.conv_support + frac_v);
                for (std::size_t sup_u = 0; sup_u < conv_full_support; ++sup_u) {
		    uvw_base_type conv_u = convolve(sup_u - (uvw_base_type)params.conv_support + frac_u);
		    convolution_base_type conv_weight = conv_v * conv_u;
                    typename active_correlation_gridding_policy::active_trait::vis_type convolved_vis = vis * conv_weight;
		    active_correlation_gridding_policy::grid_visibility(facet_output_buffer,
                        grid_size_in_floats,
                        params.nx,
                        channel_grid_index,
                        params.number_of_polarization_terms_being_gridded,
                        disc_grid_u + sup_u,
                        disc_grid_v + sup_v,
                        convolved_vis);
		    normalization_term += conv_weight;
                }
            }
	}
    }
};
}