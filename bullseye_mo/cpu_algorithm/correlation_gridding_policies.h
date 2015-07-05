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
#include "correlation_gridding_traits.h"
#include "gridding_parameters.h"
#include "cu_basic_complex.h"
#include "cu_vec.h"
namespace imaging {
  template <typename correlation_gridding_mode>
  class correlation_gridding_policy {
  public:
    typedef correlation_gridding_traits<correlation_gridding_mode> active_trait;
    static void read_corralation_data (gridding_parameters & params,
						  size_t row_index,
						  size_t spw,
						  size_t c,
						  typename active_trait::vis_type & vis,
						  typename active_trait::vis_flag_type & flag,
						  typename active_trait::vis_weight_type & weight
						 );
    static void read_channel_grid_index(const gridding_parameters & params,
						   size_t spw_channel_flat_index,
						   size_t & out);
    static void read_and_apply_antenna_jones_terms(const gridding_parameters & params,
							      size_t row_index,
							      typename active_trait::vis_type & vis);
    static void compute_facet_grid_ptr(const gridding_parameters & params,
						  size_t facet_id,
						  size_t grid_size_in_floats,
						  grid_base_type ** facet_grid_starting_ptr);
    static size_t compute_grid_offset(const gridding_parameters & params,
					   size_t grid_channel_id,
					   size_t grid_size_in_floats);
    static void grid_visibility (grid_base_type* grid,
					    size_t slice_size,
					    size_t nx,
					    size_t grid_channel_id,
					    size_t no_polarizations_being_gridded,
					    size_t pos_u,
					    size_t pos_v,
					    typename active_trait::accumulator_type & accumulator
					   );
    typedef struct avx_vis_type {} avx_vis_type;
    static void grid_visibility (grid_base_type* grid,
					    size_t slice_size,
					    size_t nx,
					    size_t pos_u,
					    size_t pos_v,
					    avx_vis_type accumulator
					   );
    static void store_normalization_term(gridding_parameters & params,std::size_t channel_grid_index,std::size_t facet_id, 
						    typename active_trait::normalization_accumulator_type normalization_weight);
  };
  /**
   * Gridding a single correlation on the CPU
   */
  template <>
  class correlation_gridding_policy<grid_single_correlation> {
  public:
    typedef correlation_gridding_traits<grid_single_correlation> active_trait;
    static void read_corralation_data (gridding_parameters & params,
						  size_t row_index,
						  size_t spw,
						  size_t c,
						  typename active_trait::vis_type & vis,
						  typename active_trait::vis_flag_type & flag,
						  typename active_trait::vis_weight_type & weight
						 ){
      size_t vis_index = (row_index * params.channel_count + c) * params.number_of_polarization_terms + params.polarization_index;
      flag = params.flags[vis_index];
      weight = params.visibility_weights[vis_index];
      vis = ((active_trait::vis_type *)params.visibilities)[vis_index];
    }
    static void read_channel_grid_index(const gridding_parameters & params,
						   size_t spw_channel_flat_index,
						   size_t & out){
      out = params.channel_grid_indicies[spw_channel_flat_index];
    }
    static void read_and_apply_antenna_jones_terms(const gridding_parameters & params,
							      size_t row_index,
							      size_t direction_id,
							      size_t spw_id,
							      size_t channel_id,
							      typename active_trait::vis_type & vis){}
    static void compute_facet_grid_ptr(const gridding_parameters & params,
						  size_t facet_id,
						  size_t grid_size_in_floats,
						  grid_base_type ** facet_grid_starting_ptr){
      *facet_grid_starting_ptr = (grid_base_type*)params.output_buffer + grid_size_in_floats * 
				params.number_of_polarization_terms_being_gridded * params.cube_channel_dim_size * facet_id;
    }
    static size_t compute_grid_offset(const gridding_parameters & params,
				    size_t grid_channel_id,
				    size_t grid_size_in_floats){
      return (grid_channel_id * params.number_of_polarization_terms_being_gridded) * grid_size_in_floats;
    }
    static inline void grid_visibility (grid_base_type* grid,
					    size_t slice_size,
					    size_t nx,
					    size_t pos_u,
					    size_t pos_v,
					    typename active_trait::accumulator_type & accumulator
					   ){
      grid_base_type* grid_flat_index = grid + 
					((pos_v * nx + pos_u) << 1);
      grid_flat_index[0] += accumulator._x._real;
      grid_flat_index[1] += accumulator._x._imag;
    }
#ifdef __AVX__
#pragma message("Compiling single correlation AVX gridding instructions")
#ifdef BULLSEYE_SINGLE
    typedef __m256 avx_vis_type[1]  __attribute__((aligned(16)));
    static inline void grid_visibility (grid_base_type* grid,
					    size_t slice_size,
					    size_t nx,
					    size_t pos_u,
					    size_t pos_v,
					    avx_vis_type accumulator
					   ){
      grid_base_type* grid_flat_index = grid + 
					((pos_v * nx + pos_u) << 1);
	 _mm256_storeu_ps(&grid_flat_index[0],
			  _mm256_add_ps(_mm256_loadu_ps(&grid_flat_index[0]),
					accumulator[0]));
    }
#elif BULLSEYE_DOUBLE
    typedef __m256d avx_vis_type[2]  __attribute__((aligned(16)));
    static inline void grid_visibility (grid_base_type* grid,
					    size_t slice_size,
					    size_t nx,
					    size_t pos_u,
					    size_t pos_v,
					    avx_vis_type accumulator
					   ){
      grid_base_type* grid_flat_index = grid + 
					((pos_v * nx + pos_u) << 1);
	 _mm256_storeu_pd(&grid_flat_index[0],
			  _mm256_add_pd(_mm256_loadu_pd(&grid_flat_index[0]),
					accumulator[0]));
	 _mm256_storeu_pd(&grid_flat_index[4],
			  _mm256_add_pd(_mm256_loadu_pd(&grid_flat_index[4]),
					accumulator[1]));
    }
#endif
#endif
    static void store_normalization_term(gridding_parameters & params,std::size_t channel_grid_index,std::size_t facet_id, 
						    typename active_trait::normalization_accumulator_type normalization_weight){
      std::size_t channel_norm_term_flat_index = facet_id * params.cube_channel_dim_size + channel_grid_index;
      params.normalization_terms[channel_norm_term_flat_index] += normalization_weight._x;
    }
  };
  /**
   * Gridding sampling function on the CPU
   */
  template <>
  class correlation_gridding_policy<grid_sampling_function>{
  public:
    typedef correlation_gridding_traits<grid_single_correlation> active_trait;
    static void read_corralation_data (gridding_parameters & params,
						  size_t row_index,
						  size_t spw,
						  size_t c,
						  typename active_trait::vis_type & vis,
						  typename active_trait::vis_flag_type & flag,
						  typename active_trait::vis_weight_type & weight
						 ){
      flag = false;
      weight = 1;
      vis = vec1<basic_complex<visibility_base_type> >(basic_complex<visibility_base_type>(1,0));
    }
    static void read_channel_grid_index(const gridding_parameters & params,
						   size_t spw_channel_flat_index,
						   size_t & out){
      out = params.sampling_function_channel_grid_indicies[spw_channel_flat_index];
    }
    static void read_and_apply_antenna_jones_terms(const gridding_parameters & params,
							      size_t row_index,
							      size_t direction_id,
							      size_t spw_id,
							      size_t channel_id,
							      typename active_trait::vis_type & vis){}
    static void compute_facet_grid_ptr(const gridding_parameters & params,
						  size_t facet_id,
						  size_t grid_size_in_floats,
						  grid_base_type ** facet_grid_starting_ptr){
      *facet_grid_starting_ptr = (grid_base_type*)params.sampling_function_buffer + grid_size_in_floats * 
				params.sampling_function_channel_count * facet_id;
    }
    static size_t compute_grid_offset(const gridding_parameters & params,
				      size_t grid_channel_id,
				      size_t grid_size_in_floats){
      return grid_channel_id * grid_size_in_floats;
    }
    static void grid_visibility (grid_base_type* grid,
					    size_t slice_size,
					    size_t nx,
					    size_t pos_u,
					    size_t pos_v,
					    typename active_trait::accumulator_type & accumulator
					   ){
      grid_base_type* grid_flat_index = grid + 
					((pos_v * nx + pos_u) << 1);
      grid_flat_index[0]+=accumulator._x._real;
      grid_flat_index[1]+=accumulator._x._imag;
    }
    static void store_normalization_term(gridding_parameters & params,std::size_t channel_grid_index,std::size_t facet_id, 
						    typename active_trait::normalization_accumulator_type normalization_weight){
      //No need to store the normalization term: centre of PSF should always be 1+0i
    }
  };
  /**
   * Gridding duel correlation on the CPU
   */
  template <>
  class correlation_gridding_policy<grid_duel_correlation> {
  public:
    typedef correlation_gridding_traits<grid_duel_correlation> active_trait;
    static void read_corralation_data (gridding_parameters & params,
						  size_t row_index,
						  size_t spw,
						  size_t c,
						  typename active_trait::vis_type & vis,
						  typename active_trait::vis_flag_type & flag,
						  typename active_trait::vis_weight_type & weight
						 ){
      size_t vis_index = (row_index * params.channel_count + c) * params.number_of_polarization_terms;
      flag._x = params.flags[vis_index + params.polarization_index];
      flag._y = params.flags[vis_index + params.second_polarization_index];
      weight._x = (params.visibility_weights)[vis_index + params.polarization_index];
      weight._y = (params.visibility_weights)[vis_index + params.second_polarization_index];
      vis._x = ((basic_complex<visibility_base_type>*)params.visibilities)[vis_index + params.polarization_index];
      vis._y = ((basic_complex<visibility_base_type>*)params.visibilities)[vis_index + params.second_polarization_index];
    }
    static void read_channel_grid_index(const gridding_parameters & params,
						   size_t spw_channel_flat_index,
						   size_t & out){
      out = params.channel_grid_indicies[spw_channel_flat_index];
    }
    static void read_and_apply_antenna_jones_terms(const gridding_parameters & params,
							      size_t row_index,
							      size_t direction_id,
							      size_t spw_id,
							      size_t channel_id,
							      typename active_trait::vis_type & vis){}
    static void compute_facet_grid_ptr(const gridding_parameters & params,
						  size_t facet_id,
						  size_t grid_size_in_floats,
						  grid_base_type ** facet_grid_starting_ptr){
      *facet_grid_starting_ptr = (grid_base_type*)params.output_buffer + grid_size_in_floats * 
				params.number_of_polarization_terms_being_gridded * params.cube_channel_dim_size * facet_id;
    }
    static size_t compute_grid_offset(const gridding_parameters & params,
				    size_t grid_channel_id,
				    size_t grid_size_in_floats){
      return (grid_channel_id * params.number_of_polarization_terms_being_gridded) * grid_size_in_floats;
    }
    static void grid_visibility (grid_base_type* grid,
					    size_t slice_size,
					    size_t nx,
					    size_t pos_u,
					    size_t pos_v,
					    typename active_trait::accumulator_type & accumulator
					   ){
      //duel correlation grids (no_facets * no_channel_averaging_grids * no_correlations * ny * nx * 2)
      grid_base_type* grid_flat_index_corr1 = grid + 
					      ((pos_v * nx + pos_u) << 1);
      grid_base_type* grid_flat_index_corr2 = grid_flat_index_corr1 + slice_size;
      grid_flat_index_corr1[0] += accumulator._x._real;
      grid_flat_index_corr1[1] += accumulator._x._imag;
      grid_flat_index_corr2[0] += accumulator._y._real;
      grid_flat_index_corr2[1] += accumulator._y._imag;
    }
#ifdef __AVX__
#pragma message("Compiling duel correlation AVX gridding instructions")
#ifdef BULLSEYE_SINGLE
    typedef __m256 avx_vis_type[2]  __attribute__((aligned(16)));
    static inline void grid_visibility (grid_base_type* grid,
					    size_t slice_size,
					    size_t nx,
					    size_t pos_u,
					    size_t pos_v,
					    avx_vis_type accumulator
					   ){
	 grid_base_type* grid_flat_index_corr1 = grid + 
					      ((pos_v * nx + pos_u) << 1);
	 _mm256_storeu_ps(&grid_flat_index_corr1[0],
			  _mm256_add_ps(_mm256_loadu_ps(&grid_flat_index_corr1[0]),
					accumulator[0]));
	 grid_base_type* grid_flat_index_corr2 = grid_flat_index_corr1 + slice_size;
	 _mm256_storeu_ps(&grid_flat_index_corr2[0],
			  _mm256_add_ps(_mm256_loadu_ps(&grid_flat_index_corr2[0]),
					accumulator[1]));
    }
#elif BULLSEYE_DOUBLE
    typedef __m256d avx_vis_type[4]  __attribute__((aligned(16)));
    static inline void grid_visibility (grid_base_type* grid,
					    size_t slice_size,
					    size_t nx,
					    size_t pos_u,
					    size_t pos_v,
					    avx_vis_type accumulator
					   ){
      grid_base_type* grid_flat_index_corr1 = grid + 
					      ((pos_v * nx + pos_u) << 1);
      _mm256_storeu_pd(&grid_flat_index_corr1[0],
		      _mm256_add_pd(_mm256_loadu_pd(&grid_flat_index_corr1[0]),
				    accumulator[0]));
      _mm256_storeu_pd(&grid_flat_index_corr1[4],
		      _mm256_add_pd(_mm256_loadu_pd(&grid_flat_index_corr1[4]),
				    accumulator[1]));
      grid_base_type* grid_flat_index_corr2 = grid_flat_index_corr1 + slice_size;
      _mm256_storeu_pd(&grid_flat_index_corr2[0],
		      _mm256_add_pd(_mm256_loadu_pd(&grid_flat_index_corr2[0]),
				    accumulator[2]));
      _mm256_storeu_pd(&grid_flat_index_corr2[4],
		      _mm256_add_pd(_mm256_loadu_pd(&grid_flat_index_corr2[4]),
				    accumulator[3]));
    }
#endif
#endif
    static void store_normalization_term(gridding_parameters & params,std::size_t channel_grid_index,std::size_t facet_id, 
						    typename active_trait::normalization_accumulator_type normalization_weight){
      std::size_t channel_norm_term_flat_index = (facet_id * params.cube_channel_dim_size + channel_grid_index) << 1;
      params.normalization_terms[channel_norm_term_flat_index] += normalization_weight._x;
      params.normalization_terms[channel_norm_term_flat_index + 1] += normalization_weight._y;
    }
  };
  template <>
  class correlation_gridding_policy<grid_4_correlation> {
  public:
    typedef correlation_gridding_traits<grid_4_correlation> active_trait;
    static void read_corralation_data (gridding_parameters & params,
						  size_t row_index,
						  size_t spw,
						  size_t c,
						  typename active_trait::vis_type & vis,
						  typename active_trait::vis_flag_type & flag,
						  typename active_trait::vis_weight_type & weight
						 ){
      size_t vis_index = (row_index * params.channel_count + c);
      //read out 4 terms at a time:
      flag = ((active_trait::vis_flag_type *)params.flags)[vis_index];
      weight = ((active_trait::vis_weight_type *)params.visibility_weights)[vis_index];
      vis = ((active_trait::vis_type *)params.visibilities)[vis_index];
    }
    static void read_channel_grid_index(const gridding_parameters & params,
						   size_t spw_channel_flat_index,
						   size_t & out){
      out = params.channel_grid_indicies[spw_channel_flat_index];
    }
    static void read_and_apply_antenna_jones_terms(const gridding_parameters & params,
							      size_t row_index,
							      size_t direction_id,
							      size_t spw_id,
							      size_t channel_id,
							      typename active_trait::vis_type & vis){}
    static void compute_facet_grid_ptr(const gridding_parameters & params,
						  size_t facet_id,
						  size_t grid_size_in_floats,
						  grid_base_type ** facet_grid_starting_ptr){
      *facet_grid_starting_ptr = (grid_base_type*)params.output_buffer + grid_size_in_floats * 
				params.number_of_polarization_terms_being_gridded * params.cube_channel_dim_size * facet_id;
    }
    static size_t compute_grid_offset(const gridding_parameters & params,
				    size_t grid_channel_id,
				    size_t grid_size_in_floats){
      return (grid_channel_id * params.number_of_polarization_terms_being_gridded) * grid_size_in_floats;
    }
    static void grid_visibility (grid_base_type* grid,
					    size_t slice_size,
					    size_t nx,
					    size_t pos_u,
					    size_t pos_v,
					    typename active_trait::accumulator_type & accumulator
					   ){
      //duel correlation grids (no_facets * no_channel_averaging_grids * no_correlations * ny * nx * 2)
      grid_base_type* grid_flat_index_corr1 = grid + 
					      ((pos_v * nx + pos_u) << 1);
      grid_base_type* grid_flat_index_corr2 = grid_flat_index_corr1 + slice_size;
      grid_base_type* grid_flat_index_corr3 = grid_flat_index_corr2 + slice_size;
      grid_base_type* grid_flat_index_corr4 = grid_flat_index_corr3 + slice_size;
      grid_flat_index_corr1[0]+=accumulator._x._real;
      grid_flat_index_corr1[1]+=accumulator._x._imag;
      grid_flat_index_corr2[0]+=accumulator._y._real;
      grid_flat_index_corr2[1]+=accumulator._y._imag;
      grid_flat_index_corr3[0]+=accumulator._z._real;
      grid_flat_index_corr3[1]+=accumulator._z._imag;
      grid_flat_index_corr4[0]+=accumulator._w._real;
      grid_flat_index_corr4[1]+=accumulator._w._imag;
    }
#ifdef __AVX__
#pragma message("Compiling quad correlation AVX gridding instructions")
#ifdef BULLSEYE_SINGLE
    typedef __m256 avx_vis_type[2]  __attribute__((aligned(16)));
    static inline void grid_visibility (grid_base_type* grid,
					    size_t slice_size,
					    size_t nx,
					    size_t pos_u,
					    size_t pos_v,
					    avx_vis_type accumulator
					   ){
	 grid_base_type* grid_flat_index_corr1 = grid + 
					      ((pos_v * nx + pos_u) << 1);
	 grid_base_type* grid_flat_index_corr2 = grid_flat_index_corr1 + slice_size;
	 grid_base_type* grid_flat_index_corr3 = grid_flat_index_corr2 + slice_size;
	 grid_base_type* grid_flat_index_corr4 = grid_flat_index_corr3 + slice_size;
	 _mm256_storeu_ps(&grid_flat_index_corr1[0],
			  _mm256_add_ps(_mm256_loadu_ps(&grid_flat_index_corr1[0]),
					accumulator[0]));
	 _mm256_storeu_ps(&grid_flat_index_corr2[0],
			  _mm256_add_ps(_mm256_loadu_ps(&grid_flat_index_corr2[0]),
					accumulator[1]));
	 _mm256_storeu_ps(&grid_flat_index_corr3[0],
			  _mm256_add_ps(_mm256_loadu_ps(&grid_flat_index_corr3[0]),
					accumulator[2]));
	 _mm256_storeu_ps(&grid_flat_index_corr4[0],
			  _mm256_add_ps(_mm256_loadu_ps(&grid_flat_index_corr4[0]),
					accumulator[3]));
    }
#elif BULLSEYE_DOUBLE
    typedef __m256d avx_vis_type[4]  __attribute__((aligned(16)));
    static inline void grid_visibility (grid_base_type* grid,
					    size_t slice_size,
					    size_t nx,
					    size_t pos_u,
					    size_t pos_v,
					    avx_vis_type accumulator
					   ){
      grid_base_type* grid_flat_index_corr1 = grid + 
					  ((pos_v * nx + pos_u) << 1);
      grid_base_type* grid_flat_index_corr2 = grid_flat_index_corr1 + slice_size;
      grid_base_type* grid_flat_index_corr3 = grid_flat_index_corr2 + slice_size;
      grid_base_type* grid_flat_index_corr4 = grid_flat_index_corr3 + slice_size;
      _mm256_storeu_pd(&grid_flat_index_corr1[0],
		      _mm256_add_pd(_mm256_loadu_pd(&grid_flat_index_corr1[0]),
				    accumulator[0]));
      _mm256_storeu_pd(&grid_flat_index_corr1[4],
		      _mm256_add_pd(_mm256_loadu_pd(&grid_flat_index_corr1[4]),
				    accumulator[1]));
      _mm256_storeu_pd(&grid_flat_index_corr2[0],
		      _mm256_add_pd(_mm256_loadu_pd(&grid_flat_index_corr2[0]),
				    accumulator[2]));
      _mm256_storeu_pd(&grid_flat_index_corr2[4],
		      _mm256_add_pd(_mm256_loadu_pd(&grid_flat_index_corr2[4]),
				    accumulator[3]));
      _mm256_storeu_pd(&grid_flat_index_corr3[0],
		      _mm256_add_pd(_mm256_loadu_pd(&grid_flat_index_corr3[0]),
				    accumulator[4]));
      _mm256_storeu_pd(&grid_flat_index_corr3[4],
		      _mm256_add_pd(_mm256_loadu_pd(&grid_flat_index_corr3[4]),
				    accumulator[5]));
      _mm256_storeu_pd(&grid_flat_index_corr4[0],
		      _mm256_add_pd(_mm256_loadu_pd(&grid_flat_index_corr4[0]),
				    accumulator[6]));
      _mm256_storeu_pd(&grid_flat_index_corr4[4],
		      _mm256_add_pd(_mm256_loadu_pd(&grid_flat_index_corr4[4]),
				    accumulator[7]));
    }
#endif
#endif
    static void store_normalization_term(gridding_parameters & params,std::size_t channel_grid_index,std::size_t facet_id, 
						    typename active_trait::normalization_accumulator_type normalization_weight){
      std::size_t channel_norm_term_flat_index = (facet_id * params.cube_channel_dim_size + channel_grid_index) << 2;
      params.normalization_terms[channel_norm_term_flat_index] += normalization_weight._x;
      params.normalization_terms[channel_norm_term_flat_index + 1] += normalization_weight._y;
      params.normalization_terms[channel_norm_term_flat_index + 2] += normalization_weight._z;
      params.normalization_terms[channel_norm_term_flat_index + 3] += normalization_weight._w;
    }
  };
  template <>
  class correlation_gridding_policy<grid_4_correlation_with_jones_corrections> {
  public:
    typedef correlation_gridding_traits<grid_4_correlation_with_jones_corrections> active_trait;
    static void read_corralation_data (gridding_parameters & params,
						  size_t row_index,
						  size_t spw,
						  size_t c,
						  typename active_trait::vis_type & vis,
						  typename active_trait::vis_flag_type & flag,
						  typename active_trait::vis_weight_type & weight
						 ){
	imaging::correlation_gridding_policy<grid_4_correlation>::read_corralation_data(params,row_index,
											spw,c,vis,flag,weight);
    }
    static void read_channel_grid_index(const gridding_parameters & params,
						   size_t spw_channel_flat_index,
						   size_t & out){
	imaging::correlation_gridding_policy<grid_4_correlation>::read_channel_grid_index(params,spw_channel_flat_index,out);
    }
    static void read_and_apply_antenna_jones_terms(const gridding_parameters & params,
							      size_t row_index,
							      size_t direction_id,
							      size_t spw_id,
							      size_t channel_id,
							      typename active_trait::vis_type & vis){
	size_t correlator_timestamp_id = params.timestamp_ids[row_index]; //this may / may not correspond to the time loop (depending on how many timestamps are missing per baseline)
	size_t antenna_1_id = params.antenna_1_ids[row_index];
	size_t antenna_2_id = params.antenna_2_ids[row_index]; //necessary to make sure we apply the matricies in the right order
	
	size_t antenna_1_jones_terms_flat_index = (((correlator_timestamp_id*params.antenna_count + antenna_1_id)*params.num_facet_centres + 
						    direction_id)*params.spw_count + spw_id)*params.channel_count + channel_id;
	size_t antenna_2_jones_terms_flat_index = (((correlator_timestamp_id*params.antenna_count + antenna_2_id)*params.num_facet_centres + 
						    direction_id)*params.spw_count + spw_id)*params.channel_count + channel_id;
	//these should be inverted prior to gridding
	jones_2x2<visibility_base_type> p_inv = ((jones_2x2<visibility_base_type> *) params.jones_terms)[antenna_1_jones_terms_flat_index];
	jones_2x2<visibility_base_type> q_inv = ((jones_2x2<visibility_base_type> *) params.jones_terms)[antenna_2_jones_terms_flat_index];
	imaging::do_hermitian_transpose(q_inv); // we can either invert and then take the hermitian transpose or take the hermitian transpose and then invert
	vis = p_inv * (vis * q_inv); //remember matricies don't commute!
    }
    static void compute_facet_grid_ptr(const gridding_parameters & params,
						  size_t facet_id,
						  size_t grid_size_in_floats,
						  grid_base_type ** facet_grid_starting_ptr){
	imaging::correlation_gridding_policy<grid_4_correlation>::compute_facet_grid_ptr(params,facet_id,grid_size_in_floats,facet_grid_starting_ptr);
    }
    static size_t compute_grid_offset(const gridding_parameters & params,
				    size_t grid_channel_id,
				    size_t grid_size_in_floats){
      return imaging::correlation_gridding_policy<grid_4_correlation>::compute_grid_offset(params,grid_channel_id,grid_size_in_floats);
    }
    static void grid_visibility (grid_base_type* grid,
					    size_t slice_size,
					    size_t nx,
					    size_t pos_u,
					    size_t pos_v,
					    typename active_trait::accumulator_type & accumulator
					   ){
	imaging::correlation_gridding_policy<grid_4_correlation>::grid_visibility(grid,slice_size,nx,
										  pos_u,pos_v,accumulator);
    }
#ifdef __AVX__
#pragma message("Compiling quad correlation with jones corrections AVX gridding instructions")
#ifdef BULLSEYE_SINGLE
    typedef __m256 avx_vis_type[2]  __attribute__((aligned(16)));
    static inline void grid_visibility (grid_base_type* grid,
					    size_t slice_size,
					    size_t nx,
					    size_t pos_u,
					    size_t pos_v,
					    avx_vis_type accumulator
					   ){
	imaging::correlation_gridding_policy<grid_4_correlation>::grid_visibility(grid,slice_size,nx,
										pos_u,pos_v,accumulator); 
    }
#elif BULLSEYE_DOUBLE
    typedef __m256d avx_vis_type[4]  __attribute__((aligned(16)));
    static inline void grid_visibility (grid_base_type* grid,
					    size_t slice_size,
					    size_t nx,
					    size_t pos_u,
					    size_t pos_v,
					    avx_vis_type accumulator
					   ){
      imaging::correlation_gridding_policy<grid_4_correlation>::grid_visibility(grid,slice_size,nx,
										pos_u,pos_v,accumulator);
    }
#endif
#endif
    static void store_normalization_term(gridding_parameters & params,std::size_t channel_grid_index,std::size_t facet_id, 
						    typename active_trait::normalization_accumulator_type normalization_weight){
      imaging::correlation_gridding_policy<grid_4_correlation>::store_normalization_term(params,channel_grid_index,facet_id,
											 normalization_weight);
    }
  };
}