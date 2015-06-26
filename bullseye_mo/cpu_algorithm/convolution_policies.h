#pragma once
#include <x86intrin.h>
#include "uvw_coord.h"
#include "gridding_parameters.h"
#include <cmath>
#include "cu_basic_complex.h"

namespace imaging {
class convolution_analytic_AA {};
class convolution_AA_1D_precomputed {};
class convolution_AA_2D_precomputed {};
class convolution_w_projection_precomputed {};
class convolution_w_projection_precomputed_vectorized {};
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
/**
 * Simple Nearest Neighbour convolution strategy
 */
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
        std::size_t disc_grid_u = std::lrint(translated_grid_u);
        std::size_t disc_grid_v = std::lrint(translated_grid_v);
        //to reduce the interpolation error we need to take the offset from the grid centre into account when choosing a convolution weight
        long frac_u_offset = std::lrint((-uvw._u + std::lrint(uvw._u))*params.conv_oversample);
        long frac_v_offset = std::lrint((-uvw._v + std::lrint(uvw._v))*params.conv_oversample);
        //Don't you dare go over the boundary
        if (disc_grid_v + padded_conv_full_support  >= params.ny || disc_grid_u + padded_conv_full_support >= params.nx ||
                disc_grid_v >= params.ny || disc_grid_u >= params.nx) return;
        
	std::size_t conv_v = 1*params.conv_oversample + frac_v_offset;
        for (std::size_t  sup_v = 0; sup_v < conv_full_support; ++sup_v) { //remember we have a +/- frac at both ends of the filter
            convolution_base_type conv_v_weight = params.conv[conv_v];
	    std::size_t conv_u = 1*params.conv_oversample + frac_u_offset;
            for (std::size_t sup_u = 0; sup_u < conv_full_support; ++sup_u) { //remember we have a +/- frac at both ends of the filter	      	      
	      convolution_base_type conv_u_weight = params.conv[conv_u];
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
	    }
	    conv_v += params.conv_oversample;
        } //conv_v
    }
};
/**
 * This is a simple 2D precomputed AA kernel
 */
template <typename active_correlation_gridding_policy>
class convolution_policy <active_correlation_gridding_policy,convolution_AA_2D_precomputed> {
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
	//account for interpolation error (we select the closest sample from the oversampled convolution filter)
        uvw_base_type translated_grid_u = uvw._u + grid_centre_offset_x;
        uvw_base_type translated_grid_v = uvw._v + grid_centre_offset_y;
        std::size_t  disc_grid_u = std::lrint(translated_grid_u);
        std::size_t  disc_grid_v = std::lrint(translated_grid_v);
        //to reduce the interpolation error we need to take the offset from the grid centre into account when choosing a convolution weight
        long frac_u_offset = std::lrint((-uvw._u + std::lrint(uvw._u)) * params.conv_oversample);
        long frac_v_offset = std::lrint((-uvw._v + std::lrint(uvw._v)) * params.conv_oversample);
        
	std::size_t conv_dim_size = padded_conv_full_support + (padded_conv_full_support - 1) * (params.conv_oversample - 1);	
	//Don't you dare go over the boundary
        if (disc_grid_v + padded_conv_full_support >= params.ny || disc_grid_u + padded_conv_full_support >= params.nx ||
                disc_grid_v >= params.ny || disc_grid_u >= params.nx) return;
	std::size_t conv_v = 1 * params.conv_oversample + frac_v_offset;
	for (std::size_t sup_v = 0; sup_v < conv_full_support; ++sup_v){
	  std::size_t conv_u = 1 * params.conv_oversample + frac_u_offset;
	  for (std::size_t sup_u = 0; sup_u < conv_full_support; ++sup_u){
	      std::size_t conv_flat_index = conv_v * conv_dim_size + conv_u;
	      convolution_base_type conv_weight = ((convolution_base_type *)params.conv)[conv_flat_index];
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
	  }
	  conv_v += params.conv_oversample;
	}
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
/**
 * This is a simple 2D w-projection kernel
 */
template <typename active_correlation_gridding_policy>
class convolution_policy <active_correlation_gridding_policy,convolution_w_projection_precomputed> {
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
        //W should be positive (either we grid the visibility or its conjugate baseline):	
	if (uvw._w < 0){
	  conj<visibility_base_type>(vis);
	  uvw._u *= -1;
	  uvw._v *= -1;
	  uvw._w *= -1;
	}
	//account for interpolation error (we select the closest sample from the oversampled convolution filter)
        uvw_base_type translated_grid_u = uvw._u + grid_centre_offset_x;
        uvw_base_type translated_grid_v = uvw._v + grid_centre_offset_y;
        std::size_t  disc_grid_u = std::lrint(translated_grid_u);
        std::size_t  disc_grid_v = std::lrint(translated_grid_v);
        //to reduce the interpolation error we need to take the offset from the grid centre into account when choosing a convolution weight
        long frac_u_offset = std::lrint((-uvw._u + std::lrint(uvw._u)) * params.conv_oversample);
        long frac_v_offset = std::lrint((-uvw._v + std::lrint(uvw._v)) * params.conv_oversample);
        
	std::size_t conv_dim_size = padded_conv_full_support + (padded_conv_full_support - 1) * (params.conv_oversample - 1);
	std::size_t best_fit_w_plane = std::lrint(abs(uvw._w)/(float)params.wmax_est*(params.wplanes-1));
	std::size_t filter_offset = best_fit_w_plane * conv_dim_size * conv_dim_size;
	
	//Don't you dare go over the boundary
        if (disc_grid_v + padded_conv_full_support >= params.ny || disc_grid_u + padded_conv_full_support >= params.nx ||
                disc_grid_v >= params.ny || disc_grid_u >= params.nx || best_fit_w_plane >= params.wplanes) return;
	std::size_t conv_v = filter_offset + (1 * params.conv_oversample + frac_v_offset) * conv_dim_size;
	for (std::size_t sup_v = 0; sup_v < conv_full_support; ++sup_v){
	  std::size_t conv_u = 1 * params.conv_oversample + frac_u_offset;
	  for (std::size_t sup_u = 0; sup_u < conv_full_support; ++sup_u){
	      std::size_t conv_flat_index = conv_v + conv_u;
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
	      normalization_term += conv_weight._real; // real and imaginary components roughly similar
	      conv_u += params.conv_oversample;
	  }
	  conv_v += params.conv_oversample * conv_dim_size;
	}
    }
};

/**
 * This is an AVX-vectorized 2D w-projection kernel
 */
#ifdef __AVX__
template <typename active_correlation_gridding_policy>
class convolution_policy <active_correlation_gridding_policy,convolution_w_projection_precomputed_vectorized> {
private:
   inline static void mul_vis_with_conv_weights(const vec1< basic_complex<float> > & vis_in, 
						basic_complex<convolution_base_type> conv_weight[4], 
						typename active_correlation_gridding_policy::avx_vis_type visses_out){
    //Do 4 complex multiplications using intrinsics
    convolution_base_type min_vis_i = -vis_in._x._imag;
    __m256 vis_ri_4 = _mm256_set_ps(vis_in._x._imag,vis_in._x._real,vis_in._x._imag,vis_in._x._real,
				    vis_in._x._imag,vis_in._x._real,vis_in._x._imag,vis_in._x._real);
    __m256 vis_mir_4 = _mm256_set_ps(vis_in._x._real,min_vis_i,vis_in._x._real,min_vis_i,
				    vis_in._x._real,min_vis_i,vis_in._x._real,min_vis_i);
    visses_out[0] = _mm256_add_ps(_mm256_mul_ps(vis_ri_4,
						_mm256_set_ps(conv_weight[3]._real,conv_weight[3]._real,
							      conv_weight[2]._real,conv_weight[2]._real,
							      conv_weight[1]._real,conv_weight[1]._real,
							      conv_weight[0]._real,conv_weight[0]._real)),
				   _mm256_mul_ps(vis_mir_4,
						_mm256_set_ps(conv_weight[3]._imag,conv_weight[3]._imag,
							      conv_weight[2]._imag,conv_weight[2]._imag,
							      conv_weight[1]._imag,conv_weight[1]._imag,
							      conv_weight[0]._imag,conv_weight[0]._imag)));
  }
   inline static void mul_vis_with_conv_weights(const vec1< basic_complex<double> > & vis_in, 
						basic_complex<convolution_base_type> conv_weight[4], 
						typename active_correlation_gridding_policy::avx_vis_type visses_out){
    //Do 4 complex multiplications using intrinsics
    convolution_base_type min_vis_i = -vis_in._x._imag;
    __m256d vis_ri_2 = _mm256_set_pd(vis_in._x._imag,vis_in._x._real,vis_in._x._imag,vis_in._x._real);
    __m256d vis_mir_2 = _mm256_set_pd(vis_in._x._real,min_vis_i,vis_in._x._real,min_vis_i);
    visses_out[0] = _mm256_add_pd(_mm256_mul_pd(vis_ri_2,
						_mm256_set_pd(conv_weight[1]._real,conv_weight[1]._real,
							      conv_weight[0]._real,conv_weight[0]._real)),
				  _mm256_mul_pd(vis_mir_2,
						_mm256_set_pd(conv_weight[1]._imag,conv_weight[1]._imag,
							      conv_weight[0]._imag,conv_weight[0]._imag)));
    visses_out[1] = _mm256_add_pd(_mm256_mul_pd(vis_ri_2,
						_mm256_set_pd(conv_weight[3]._real,conv_weight[3]._real,
							      conv_weight[2]._real,conv_weight[2]._real)),
				  _mm256_mul_pd(vis_mir_2,
						_mm256_set_pd(conv_weight[3]._imag,conv_weight[3]._imag,
							      conv_weight[2]._imag,conv_weight[2]._imag)));
  }
  inline static void mul_vis_with_conv_weights(const vec2< basic_complex<float> > & vis_in, 
						basic_complex<convolution_base_type> conv_weight[4], 
						typename active_correlation_gridding_policy::avx_vis_type visses_out){
    //Do 4 complex multiplications using intrinsics
    {
      convolution_base_type min_vis_i = -vis_in._x._imag;
      __m256 vis_ri_4 = _mm256_set_ps(vis_in._x._imag,vis_in._x._real,vis_in._x._imag,vis_in._x._real,
				      vis_in._x._imag,vis_in._x._real,vis_in._x._imag,vis_in._x._real);
      __m256 vis_mir_4 = _mm256_set_ps(vis_in._x._real,min_vis_i,vis_in._x._real,min_vis_i,
				      vis_in._x._real,min_vis_i,vis_in._x._real,min_vis_i);
      visses_out[0] = _mm256_add_ps(_mm256_mul_ps(vis_ri_4,
						  _mm256_set_ps(conv_weight[3]._real,conv_weight[3]._real,
								conv_weight[2]._real,conv_weight[2]._real,
								conv_weight[1]._real,conv_weight[1]._real,
								conv_weight[0]._real,conv_weight[0]._real)),
				    _mm256_mul_ps(vis_mir_4,
						  _mm256_set_ps(conv_weight[3]._imag,conv_weight[3]._imag,
								conv_weight[2]._imag,conv_weight[2]._imag,
								conv_weight[1]._imag,conv_weight[1]._imag,
								conv_weight[0]._imag,conv_weight[0]._imag)));
    }
    //second correlation
    {
      convolution_base_type min_vis_i = -vis_in._y._imag;
      __m256 vis_ri_4 = _mm256_set_ps(vis_in._y._imag,vis_in._x._real,vis_in._y._imag,vis_in._y._real,
				      vis_in._y._imag,vis_in._x._real,vis_in._y._imag,vis_in._y._real);
      __m256 vis_mir_4 = _mm256_set_ps(vis_in._y._real,min_vis_i,vis_in._y._real,min_vis_i,
				      vis_in._y._real,min_vis_i,vis_in._y._real,min_vis_i);
      visses_out[1] = _mm256_add_ps(_mm256_mul_ps(vis_ri_4,
						  _mm256_set_ps(conv_weight[3]._real,conv_weight[3]._real,
								conv_weight[2]._real,conv_weight[2]._real,
								conv_weight[1]._real,conv_weight[1]._real,
								conv_weight[0]._real,conv_weight[0]._real)),
				    _mm256_mul_ps(vis_mir_4,
						  _mm256_set_ps(conv_weight[3]._imag,conv_weight[3]._imag,
								conv_weight[2]._imag,conv_weight[2]._imag,
								conv_weight[1]._imag,conv_weight[1]._imag,
								conv_weight[0]._imag,conv_weight[0]._imag)));
    }
  }
  inline static void mul_vis_with_conv_weights(const vec2< basic_complex<double> > & vis_in, 
						basic_complex<convolution_base_type> conv_weight[4], 
						typename active_correlation_gridding_policy::avx_vis_type visses_out){
    //Do 4 complex multiplications using intrinsics
    {
	convolution_base_type min_vis_i = -vis_in._x._imag;
	__m256d vis_ri_2 = _mm256_set_pd(vis_in._x._imag,vis_in._x._real,vis_in._x._imag,vis_in._x._real);
	__m256d vis_mir_2 = _mm256_set_pd(vis_in._x._real,min_vis_i,vis_in._x._real,min_vis_i);
	visses_out[0] = _mm256_add_pd(_mm256_mul_pd(vis_ri_2,
						    _mm256_set_pd(conv_weight[1]._real,conv_weight[1]._real,
								  conv_weight[0]._real,conv_weight[0]._real)),
				      _mm256_mul_pd(vis_mir_2,
						    _mm256_set_pd(conv_weight[1]._imag,conv_weight[1]._imag,
								  conv_weight[0]._imag,conv_weight[0]._imag)));
	visses_out[1] = _mm256_add_pd(_mm256_mul_pd(vis_ri_2,
						    _mm256_set_pd(conv_weight[3]._real,conv_weight[3]._real,
								  conv_weight[2]._real,conv_weight[2]._real)),
				      _mm256_mul_pd(vis_mir_2,
						    _mm256_set_pd(conv_weight[3]._imag,conv_weight[3]._imag,
								  conv_weight[2]._imag,conv_weight[2]._imag)));
    }
    //second correlation
    {
	convolution_base_type min_vis_i = -vis_in._y._imag;
	__m256d vis_ri_2 = _mm256_set_pd(vis_in._y._imag,vis_in._y._real,vis_in._y._imag,vis_in._x._real);
	__m256d vis_mir_2 = _mm256_set_pd(vis_in._y._real,min_vis_i,vis_in._y._real,min_vis_i);
	visses_out[2] = _mm256_add_pd(_mm256_mul_pd(vis_ri_2,
						    _mm256_set_pd(conv_weight[1]._real,conv_weight[1]._real,
								  conv_weight[0]._real,conv_weight[0]._real)),
				      _mm256_mul_pd(vis_mir_2,
						    _mm256_set_pd(conv_weight[1]._imag,conv_weight[1]._imag,
								  conv_weight[0]._imag,conv_weight[0]._imag)));
	visses_out[3] = _mm256_add_pd(_mm256_mul_pd(vis_ri_2,
						    _mm256_set_pd(conv_weight[3]._real,conv_weight[3]._real,
								  conv_weight[2]._real,conv_weight[2]._real)),
				      _mm256_mul_pd(vis_mir_2,
						    _mm256_set_pd(conv_weight[3]._imag,conv_weight[3]._imag,
								  conv_weight[2]._imag,conv_weight[2]._imag)));
    }
  }
  inline static void mul_vis_with_conv_weights(const vec4< basic_complex<float> > & vis_in, 
						basic_complex<convolution_base_type> conv_weight[4], 
						typename active_correlation_gridding_policy::avx_vis_type visses_out){
    //Do 4 complex multiplications using intrinsics
    {
      convolution_base_type min_vis_i = -vis_in._x._imag;
      __m256 vis_ri_4 = _mm256_set_ps(vis_in._x._imag,vis_in._x._real,vis_in._x._imag,vis_in._x._real,
				      vis_in._x._imag,vis_in._x._real,vis_in._x._imag,vis_in._x._real);
      __m256 vis_mir_4 = _mm256_set_ps(vis_in._x._real,min_vis_i,vis_in._x._real,min_vis_i,
				      vis_in._x._real,min_vis_i,vis_in._x._real,min_vis_i);
      visses_out[0] = _mm256_add_ps(_mm256_mul_ps(vis_ri_4,
						  _mm256_set_ps(conv_weight[3]._real,conv_weight[3]._real,
								conv_weight[2]._real,conv_weight[2]._real,
								conv_weight[1]._real,conv_weight[1]._real,
								conv_weight[0]._real,conv_weight[0]._real)),
				    _mm256_mul_ps(vis_mir_4,
						  _mm256_set_ps(conv_weight[3]._imag,conv_weight[3]._imag,
								conv_weight[2]._imag,conv_weight[2]._imag,
								conv_weight[1]._imag,conv_weight[1]._imag,
								conv_weight[0]._imag,conv_weight[0]._imag)));
    }
    //second correlation
    {
      convolution_base_type min_vis_i = -vis_in._y._imag;
      __m256 vis_ri_4 = _mm256_set_ps(vis_in._y._imag,vis_in._x._real,vis_in._y._imag,vis_in._y._real,
				      vis_in._y._imag,vis_in._x._real,vis_in._y._imag,vis_in._y._real);
      __m256 vis_mir_4 = _mm256_set_ps(vis_in._y._real,min_vis_i,vis_in._y._real,min_vis_i,
				      vis_in._y._real,min_vis_i,vis_in._y._real,min_vis_i);
      visses_out[1] = _mm256_add_ps(_mm256_mul_ps(vis_ri_4,
						  _mm256_set_ps(conv_weight[3]._real,conv_weight[3]._real,
								conv_weight[2]._real,conv_weight[2]._real,
								conv_weight[1]._real,conv_weight[1]._real,
								conv_weight[0]._real,conv_weight[0]._real)),
				    _mm256_mul_ps(vis_mir_4,
						  _mm256_set_ps(conv_weight[3]._imag,conv_weight[3]._imag,
								conv_weight[2]._imag,conv_weight[2]._imag,
								conv_weight[1]._imag,conv_weight[1]._imag,
								conv_weight[0]._imag,conv_weight[0]._imag)));
    }
    //third correlation
    {
      convolution_base_type min_vis_i = -vis_in._z._imag;
      __m256 vis_ri_4 = _mm256_set_ps(vis_in._z._imag,vis_in._z._real,vis_in._z._imag,vis_in._z._real,
				      vis_in._z._imag,vis_in._z._real,vis_in._z._imag,vis_in._z._real);
      __m256 vis_mir_4 = _mm256_set_ps(vis_in._z._real,min_vis_i,vis_in._z._real,min_vis_i,
				      vis_in._z._real,min_vis_i,vis_in._z._real,min_vis_i);
      visses_out[2] = _mm256_add_ps(_mm256_mul_ps(vis_ri_4,
						  _mm256_set_ps(conv_weight[3]._real,conv_weight[3]._real,
								conv_weight[2]._real,conv_weight[2]._real,
								conv_weight[1]._real,conv_weight[1]._real,
								conv_weight[0]._real,conv_weight[0]._real)),
				    _mm256_mul_ps(vis_mir_4,
						  _mm256_set_ps(conv_weight[3]._imag,conv_weight[3]._imag,
								conv_weight[2]._imag,conv_weight[2]._imag,
								conv_weight[1]._imag,conv_weight[1]._imag,
								conv_weight[0]._imag,conv_weight[0]._imag)));
    }
    //fourth correlation
    {
      convolution_base_type min_vis_i = -vis_in._y._imag;
      __m256 vis_ri_4 = _mm256_set_ps(vis_in._w._imag,vis_in._x._real,vis_in._w._imag,vis_in._w._real,
				      vis_in._w._imag,vis_in._x._real,vis_in._w._imag,vis_in._w._real);
      __m256 vis_mir_4 = _mm256_set_ps(vis_in._w._real,min_vis_i,vis_in._w._real,min_vis_i,
				      vis_in._w._real,min_vis_i,vis_in._w._real,min_vis_i);
      visses_out[3] = _mm256_add_ps(_mm256_mul_ps(vis_ri_4,
						  _mm256_set_ps(conv_weight[3]._real,conv_weight[3]._real,
								conv_weight[2]._real,conv_weight[2]._real,
								conv_weight[1]._real,conv_weight[1]._real,
								conv_weight[0]._real,conv_weight[0]._real)),
				    _mm256_mul_ps(vis_mir_4,
						  _mm256_set_ps(conv_weight[3]._imag,conv_weight[3]._imag,
								conv_weight[2]._imag,conv_weight[2]._imag,
								conv_weight[1]._imag,conv_weight[1]._imag,
								conv_weight[0]._imag,conv_weight[0]._imag)));
    }
  }
  inline static void mul_vis_with_conv_weights(const vec4< basic_complex<double> > & vis_in, 
						basic_complex<convolution_base_type> conv_weight[4], 
						typename active_correlation_gridding_policy::avx_vis_type visses_out){
    //Do 4 complex multiplications using intrinsics
    {
	convolution_base_type min_vis_i = -vis_in._x._imag;
	__m256d vis_ri_2 = _mm256_set_pd(vis_in._x._imag,vis_in._x._real,vis_in._x._imag,vis_in._x._real);
	__m256d vis_mir_2 = _mm256_set_pd(vis_in._x._real,min_vis_i,vis_in._x._real,min_vis_i);
	visses_out[0] = _mm256_add_pd(_mm256_mul_pd(vis_ri_2,
						    _mm256_set_pd(conv_weight[1]._real,conv_weight[1]._real,
								  conv_weight[0]._real,conv_weight[0]._real)),
				      _mm256_mul_pd(vis_mir_2,
						    _mm256_set_pd(conv_weight[1]._imag,conv_weight[1]._imag,
								  conv_weight[0]._imag,conv_weight[0]._imag)));
	visses_out[1] = _mm256_add_pd(_mm256_mul_pd(vis_ri_2,
						    _mm256_set_pd(conv_weight[3]._real,conv_weight[3]._real,
								  conv_weight[2]._real,conv_weight[2]._real)),
				      _mm256_mul_pd(vis_mir_2,
						    _mm256_set_pd(conv_weight[3]._imag,conv_weight[3]._imag,
								  conv_weight[2]._imag,conv_weight[2]._imag)));
    }
    //second correlation
    {
	convolution_base_type min_vis_i = -vis_in._y._imag;
	__m256d vis_ri_2 = _mm256_set_pd(vis_in._y._imag,vis_in._y._real,vis_in._y._imag,vis_in._x._real);
	__m256d vis_mir_2 = _mm256_set_pd(vis_in._y._real,min_vis_i,vis_in._y._real,min_vis_i);
	visses_out[2] = _mm256_add_pd(_mm256_mul_pd(vis_ri_2,
						    _mm256_set_pd(conv_weight[1]._real,conv_weight[1]._real,
								  conv_weight[0]._real,conv_weight[0]._real)),
				      _mm256_mul_pd(vis_mir_2,
						    _mm256_set_pd(conv_weight[1]._imag,conv_weight[1]._imag,
								  conv_weight[0]._imag,conv_weight[0]._imag)));
	visses_out[3] = _mm256_add_pd(_mm256_mul_pd(vis_ri_2,
						    _mm256_set_pd(conv_weight[3]._real,conv_weight[3]._real,
								  conv_weight[2]._real,conv_weight[2]._real)),
				      _mm256_mul_pd(vis_mir_2,
						    _mm256_set_pd(conv_weight[3]._imag,conv_weight[3]._imag,
								  conv_weight[2]._imag,conv_weight[2]._imag)));
    }
    //third correlation
    {
	convolution_base_type min_vis_i = -vis_in._z._imag;
	__m256d vis_ri_2 = _mm256_set_pd(vis_in._z._imag,vis_in._z._real,vis_in._z._imag,vis_in._z._real);
	__m256d vis_mir_2 = _mm256_set_pd(vis_in._z._real,min_vis_i,vis_in._z._real,min_vis_i);
	visses_out[4] = _mm256_add_pd(_mm256_mul_pd(vis_ri_2,
						    _mm256_set_pd(conv_weight[1]._real,conv_weight[1]._real,
								  conv_weight[0]._real,conv_weight[0]._real)),
				      _mm256_mul_pd(vis_mir_2,
						    _mm256_set_pd(conv_weight[1]._imag,conv_weight[1]._imag,
								  conv_weight[0]._imag,conv_weight[0]._imag)));
	visses_out[5] = _mm256_add_pd(_mm256_mul_pd(vis_ri_2,
						    _mm256_set_pd(conv_weight[3]._real,conv_weight[3]._real,
								  conv_weight[2]._real,conv_weight[2]._real)),
				      _mm256_mul_pd(vis_mir_2,
						    _mm256_set_pd(conv_weight[3]._imag,conv_weight[3]._imag,
								  conv_weight[2]._imag,conv_weight[2]._imag)));
    }
    //fourth correlation
    {
	convolution_base_type min_vis_i = -vis_in._w._imag;
	__m256d vis_ri_2 = _mm256_set_pd(vis_in._w._imag,vis_in._w._real,vis_in._w._imag,vis_in._w._real);
	__m256d vis_mir_2 = _mm256_set_pd(vis_in._w._real,min_vis_i,vis_in._w._real,min_vis_i);
	visses_out[6] = _mm256_add_pd(_mm256_mul_pd(vis_ri_2,
						    _mm256_set_pd(conv_weight[1]._real,conv_weight[1]._real,
								  conv_weight[0]._real,conv_weight[0]._real)),
				      _mm256_mul_pd(vis_mir_2,
						    _mm256_set_pd(conv_weight[1]._imag,conv_weight[1]._imag,
								  conv_weight[0]._imag,conv_weight[0]._imag)));
	visses_out[7] = _mm256_add_pd(_mm256_mul_pd(vis_ri_2,
						    _mm256_set_pd(conv_weight[3]._real,conv_weight[3]._real,
								  conv_weight[2]._real,conv_weight[2]._real)),
				      _mm256_mul_pd(vis_mir_2,
						    _mm256_set_pd(conv_weight[3]._imag,conv_weight[3]._imag,
								  conv_weight[2]._imag,conv_weight[2]._imag)));
    }
  }
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
        //W should be positive (either we grid the visibility or its conjugate baseline):	
	if (uvw._w < 0){
	  conj<visibility_base_type>(vis);
	  uvw._u *= -1;
	  uvw._v *= -1;
	  uvw._w *= -1;
	}
	//account for interpolation error (we select the closest sample from the oversampled convolution filter)
        uvw_base_type translated_grid_u = uvw._u + grid_centre_offset_x;
        uvw_base_type translated_grid_v = uvw._v + grid_centre_offset_y;
        std::size_t  disc_grid_u = std::lrint(translated_grid_u);
        std::size_t  disc_grid_v = std::lrint(translated_grid_v);
        //to reduce the interpolation error we need to take the offset from the grid centre into account when choosing a convolution weight
        long frac_u_offset = std::lrint((-uvw._u + std::lrint(uvw._u)) * params.conv_oversample);
        long frac_v_offset = std::lrint((-uvw._v + std::lrint(uvw._v)) * params.conv_oversample);
        
	std::size_t conv_dim_size = padded_conv_full_support + (padded_conv_full_support - 1) * (params.conv_oversample - 1);
	std::size_t best_fit_w_plane = std::lrint(abs(uvw._w)/(float)params.wmax_est*(params.wplanes-1));
	std::size_t filter_offset = best_fit_w_plane * conv_dim_size * conv_dim_size;
	
	//Don't you dare go over the boundary
        if (disc_grid_v + padded_conv_full_support >= params.ny || disc_grid_u + padded_conv_full_support >= params.nx ||
                disc_grid_v >= params.ny || disc_grid_u >= params.nx || best_fit_w_plane >= params.wplanes) return;
	std::size_t conv_v = filter_offset + (1 * params.conv_oversample + frac_v_offset) * conv_dim_size;
	std::size_t unrolled_ul = conv_full_support / 4;
	std::size_t rem_loop_ll = (unrolled_ul) * 4;
	for (std::size_t sup_v = 0; sup_v < conv_full_support; ++sup_v){
	  for (std::size_t sup_u = 0; sup_u < unrolled_ul; ++sup_u){
	      std::size_t first_conv_u = (1 + sup_u * 4) * params.conv_oversample + frac_u_offset;
	      std::size_t first_conv_flat_index = conv_v + first_conv_u;
	      basic_complex<convolution_base_type> conv_weight[4] = {((basic_complex<convolution_base_type>*)params.conv)[first_conv_flat_index],
								     ((basic_complex<convolution_base_type>*)params.conv)[first_conv_flat_index + params.conv_oversample],
								     ((basic_complex<convolution_base_type>*)params.conv)[first_conv_flat_index + params.conv_oversample * 2],
								     ((basic_complex<convolution_base_type>*)params.conv)[first_conv_flat_index + params.conv_oversample * 3]};
	      typename active_correlation_gridding_policy::avx_vis_type convolved_vis;
	      mul_vis_with_conv_weights(vis,conv_weight,convolved_vis);
	      {
		active_correlation_gridding_policy::grid_visibility(facet_output_buffer,
								    grid_size_in_floats,
								    params.nx,
								    channel_grid_index,
								    params.number_of_polarization_terms_being_gridded,
								    disc_grid_u + sup_u * 4,
								    disc_grid_v + sup_v,
								    convolved_vis);
	      }
	      normalization_term += conv_weight[0]._real + conv_weight[1]._real + conv_weight[2]._real + conv_weight[3]._real;// real and imaginary components roughly similar
	  }
	  for (std::size_t sup_u = rem_loop_ll; sup_u < conv_full_support; ++sup_u){
	      std::size_t conv_u = (1 + sup_u) * params.conv_oversample + frac_u_offset;
	      std::size_t conv_flat_index = conv_v + conv_u;
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
	      normalization_term += conv_weight._real; // real and imaginary components roughly similar
	  }
	  conv_v += params.conv_oversample * conv_dim_size;
	}
    }
};
#endif
}