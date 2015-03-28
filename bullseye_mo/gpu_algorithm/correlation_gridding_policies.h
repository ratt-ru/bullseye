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
#include "cu_common.h"
#include "cu_double_atomic.h"
namespace imaging {
  template <typename correlation_gridding_mode>
  class correlation_gridding_policy {
  public:
    typedef correlation_gridding_traits<correlation_gridding_mode> active_trait;
    __device__ static void read_corralation_data (gridding_parameters & params,
						  size_t row_index,
						  size_t spw,
						  size_t c,
						  typename active_trait::vis_type & vis,
						  typename active_trait::vis_flag_type & flag,
						  typename active_trait::vis_weight_type & weight
						 );
    __device__ static void read_channel_grid_index(const gridding_parameters & params,
						   size_t spw_channel_flat_index,
						   size_t & out);
    __device__ static void read_and_apply_antenna_jones_terms(const gridding_parameters & params,
							      size_t row_index,
							      typename active_trait::vis_type & vis);
    __device__ static void compute_facet_grid_ptr(const gridding_parameters & params,
						  size_t facet_id,
						  size_t grid_size_in_floats,
						  grid_base_type ** facet_grid_starting_ptr);
    __device__ static void grid_visibility (grid_base_type* grid,
					    size_t slice_size,
					    size_t nx,
					    size_t grid_channel_id,
					    size_t no_polarizations_being_gridded,
					    size_t pos_u,
					    size_t pos_v,
					    typename active_trait::accumulator_type & accumulator
					   );
    __device__ static void update_normalization_accumulator(const gridding_parameters & params,
							    typename active_trait::normalization_accumulator_type & accumulator,
							    size_t facet_id,
							    size_t grid_channel_id,
							    size_t convolution_full_support,
							    size_t conv_u_id,
							    size_t conv_v_id);
  };
  template <>
  class correlation_gridding_policy<grid_single_correlation> {
  public:
    typedef correlation_gridding_traits<grid_single_correlation> active_trait;
    __device__ static void read_corralation_data (gridding_parameters & params,
						  size_t row_index,
						  size_t spw,
						  size_t c,
						  typename active_trait::vis_type & vis,
						  typename active_trait::vis_flag_type & flag,
						  typename active_trait::vis_weight_type & weight
						 ){
      size_t vis_index = row_index * params.channel_count + c;
      flag = params.flags[vis_index];
      weight = params.visibility_weights[vis_index];
      vis = ((active_trait::vis_type *)params.visibilities)[vis_index];
    }
    __device__ static void read_channel_grid_index(const gridding_parameters & params,
						   size_t spw_channel_flat_index,
						   size_t & out){
      out = params.channel_grid_indicies[spw_channel_flat_index];
    }
    __device__ static void read_and_apply_antenna_jones_terms(const gridding_parameters & params,
							      size_t row_index,
							      typename active_trait::vis_type & vis){}
    __device__ static void compute_facet_grid_ptr(const gridding_parameters & params,
						  size_t facet_id,
						  size_t grid_size_in_floats,
						  grid_base_type ** facet_grid_starting_ptr){
      *facet_grid_starting_ptr = (grid_base_type*)params.output_buffer + grid_size_in_floats * 
				params.number_of_polarization_terms_being_gridded * params.cube_channel_dim_size * facet_id;
    }
    __device__ static void grid_visibility (grid_base_type* grid,
					    size_t slice_size,
					    size_t nx,
					    size_t grid_channel_id,
					    size_t no_polarizations_being_gridded,
					    size_t pos_u,
					    size_t pos_v,
					    typename active_trait::accumulator_type & accumulator
					   ){
      grid_base_type* grid_flat_index = grid + 
					(grid_channel_id * slice_size) + 
					((pos_v * nx + pos_u) << 1);
      atomicAdd(grid_flat_index,accumulator._x._real);
      atomicAdd(grid_flat_index + 1,accumulator._x._imag);
    }
    __device__ static void update_normalization_accumulator(const gridding_parameters & params,
							    typename active_trait::normalization_accumulator_type & accumulator,
							    size_t facet_id,
							    size_t grid_channel_id,
							    size_t convolution_full_support,
							    size_t conv_u_id,
							    size_t conv_v_id){
      size_t flat_index_up_to_corr = (facet_id * params.cube_channel_dim_size + grid_channel_id);
      size_t reduction_step_size = convolution_full_support * convolution_full_support;
      size_t current_reduction_step_flat_index = conv_v_id * convolution_full_support + conv_u_id;
      atomicAdd(params.normalization_terms + (flat_index_up_to_corr * reduction_step_size + current_reduction_step_flat_index),accumulator._x);
    }
  };
  template <>
  class correlation_gridding_policy<grid_sampling_function>{
  public:
    typedef correlation_gridding_traits<grid_single_correlation> active_trait;
    __device__ static void read_corralation_data (gridding_parameters & params,
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
    __device__ static void read_channel_grid_index(const gridding_parameters & params,
						   size_t spw_channel_flat_index,
						   size_t & out){
      out = params.sampling_function_channel_grid_indicies[spw_channel_flat_index];
    }
    __device__ static void read_and_apply_antenna_jones_terms(const gridding_parameters & params,
							      size_t row_index,
							      typename active_trait::vis_type & vis){}
    __device__ static void compute_facet_grid_ptr(const gridding_parameters & params,
						  size_t facet_id,
						  size_t grid_size_in_floats,
						  grid_base_type ** facet_grid_starting_ptr){
      *facet_grid_starting_ptr = (grid_base_type*)params.sampling_function_buffer + grid_size_in_floats * 
				params.sampling_function_channel_count * facet_id;
    }
    __device__ static void grid_visibility (grid_base_type* grid,
					    size_t slice_size,
					    size_t nx,
					    size_t grid_channel_id,
					    size_t no_polarizations_being_gridded,
					    size_t pos_u,
					    size_t pos_v,
					    typename active_trait::accumulator_type & accumulator
					   ){
      grid_base_type* grid_flat_index = grid + 
					(grid_channel_id * slice_size) + 
					((pos_v * nx + pos_u) << 1);
      atomicAdd(grid_flat_index,accumulator._x._real);
      atomicAdd(grid_flat_index + 1,accumulator._x._imag);
    }
    __device__ static void update_normalization_accumulator(const gridding_parameters & params,
							    typename active_trait::normalization_accumulator_type & accumulator,
							    size_t facet_id,
							    size_t grid_channel_id,
							    size_t convolution_full_support,
							    size_t conv_u_id,
							    size_t conv_v_id){
    }
  };
  template <>
  class correlation_gridding_policy<grid_duel_correlation> {
  public:
    typedef correlation_gridding_traits<grid_duel_correlation> active_trait;
    __device__ static void read_corralation_data (gridding_parameters & params,
						  size_t row_index,
						  size_t spw,
						  size_t c,
						  typename active_trait::vis_type & vis,
						  typename active_trait::vis_flag_type & flag,
						  typename active_trait::vis_weight_type & weight
						 ){
      size_t vis_index = (row_index * params.channel_count + c); //this assumes the data is stripped of excess correlations before transferred to GPU
      flag = ((active_trait::vis_flag_type *)params.flags)[vis_index];
      weight = ((active_trait::vis_weight_type *)params.visibility_weights)[vis_index];
      vis = ((active_trait::vis_type *)params.visibilities)[vis_index];
    }
    __device__ static void read_channel_grid_index(const gridding_parameters & params,
						   size_t spw_channel_flat_index,
						   size_t & out){
      out = params.channel_grid_indicies[spw_channel_flat_index];
    }
    __device__ static void read_and_apply_antenna_jones_terms(const gridding_parameters & params,
							      size_t row_index,
							      typename active_trait::vis_type & vis){}
    __device__ static void compute_facet_grid_ptr(const gridding_parameters & params,
						  size_t facet_id,
						  size_t grid_size_in_floats,
						  grid_base_type ** facet_grid_starting_ptr){
      *facet_grid_starting_ptr = (grid_base_type*)params.output_buffer + grid_size_in_floats * 
				params.number_of_polarization_terms_being_gridded * params.cube_channel_dim_size * facet_id;
    }
    __device__ static void grid_visibility (grid_base_type* grid,
					    size_t slice_size,
					    size_t nx,
					    size_t grid_channel_id,
					    size_t no_polarizations_being_gridded,
					    size_t pos_u,
					    size_t pos_v,
					    typename active_trait::accumulator_type & accumulator
					   ){
      //duel correlation grids (no_facets * no_channel_averaging_grids * no_correlations * ny * nx * 2)
      grid_base_type* grid_flat_index_corr1 = grid + 
					      ((grid_channel_id * no_polarizations_being_gridded) * slice_size) + 
					      ((pos_v * nx + pos_u) << 1);
      grid_base_type* grid_flat_index_corr2 = grid_flat_index_corr1 + slice_size;
      atomicAdd(grid_flat_index_corr1,accumulator._x._real);
      atomicAdd(grid_flat_index_corr1 + 1,accumulator._x._imag);
      atomicAdd(grid_flat_index_corr2,accumulator._y._real);
      atomicAdd(grid_flat_index_corr2 + 1,accumulator._y._imag);
    }
    __device__ static void update_normalization_accumulator(const gridding_parameters & params,
							    typename active_trait::normalization_accumulator_type & accumulator,
							    size_t facet_id,
							    size_t grid_channel_id,
							    size_t convolution_full_support,
							    size_t conv_u_id,
							    size_t conv_v_id){
      size_t flat_index_up_to_corr = (facet_id * params.cube_channel_dim_size + grid_channel_id) * params.number_of_polarization_terms_being_gridded;
      size_t reduction_step_size = convolution_full_support * convolution_full_support;
      size_t current_reduction_step_flat_index = conv_v_id * convolution_full_support + conv_u_id;
      atomicAdd(params.normalization_terms + (flat_index_up_to_corr * reduction_step_size + current_reduction_step_flat_index),accumulator._x);
      atomicAdd(params.normalization_terms + ((flat_index_up_to_corr + 1) * reduction_step_size + current_reduction_step_flat_index), accumulator._y);
    }
  };
  template <>
  class correlation_gridding_policy<grid_4_correlation> {
  public:
    typedef correlation_gridding_traits<grid_4_correlation> active_trait;
    __device__ static void read_corralation_data (gridding_parameters & params,
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
    __device__ static void read_channel_grid_index(const gridding_parameters & params,
						   size_t spw_channel_flat_index,
						   size_t & out){
      out = params.channel_grid_indicies[spw_channel_flat_index];
    }
    __device__ static void read_and_apply_antenna_jones_terms(const gridding_parameters & params,
							      size_t row_index,
							      typename active_trait::vis_type & vis){}
    __device__ static void compute_facet_grid_ptr(const gridding_parameters & params,
						  size_t facet_id,
						  size_t grid_size_in_floats,
						  grid_base_type ** facet_grid_starting_ptr){
      *facet_grid_starting_ptr = (grid_base_type*)params.output_buffer + grid_size_in_floats * 
				params.number_of_polarization_terms_being_gridded * params.cube_channel_dim_size * facet_id;
    }
    __device__ static void grid_visibility (grid_base_type* grid,
					    size_t slice_size,
					    size_t nx,
					    size_t grid_channel_id,
					    size_t no_polarizations_being_gridded,
					    size_t pos_u,
					    size_t pos_v,
					    typename active_trait::accumulator_type & accumulator
					   ){
      //duel correlation grids (no_facets * no_channel_averaging_grids * no_correlations * ny * nx * 2)
      grid_base_type* grid_flat_index_corr1 = grid + 
					      ((grid_channel_id * no_polarizations_being_gridded) * slice_size) + 
					      ((pos_v * nx + pos_u) << 1);
      grid_base_type* grid_flat_index_corr2 = grid_flat_index_corr1 + slice_size;
      grid_base_type* grid_flat_index_corr3 = grid_flat_index_corr2 + slice_size;
      grid_base_type* grid_flat_index_corr4 = grid_flat_index_corr3 + slice_size;
      atomicAdd(grid_flat_index_corr1,accumulator._x._real);
      atomicAdd(grid_flat_index_corr1 + 1,accumulator._x._imag);
      atomicAdd(grid_flat_index_corr2,accumulator._y._real);
      atomicAdd(grid_flat_index_corr2 + 1,accumulator._y._imag);
      atomicAdd(grid_flat_index_corr3,accumulator._z._real);
      atomicAdd(grid_flat_index_corr3 + 1,accumulator._z._imag);
      atomicAdd(grid_flat_index_corr4,accumulator._w._real);
      atomicAdd(grid_flat_index_corr4 + 1,accumulator._w._imag);
    }
    __device__ static void update_normalization_accumulator(const gridding_parameters & params,
							    typename active_trait::normalization_accumulator_type & accumulator,
							    size_t facet_id,
							    size_t grid_channel_id,
							    size_t convolution_full_support,
							    size_t conv_u_id,
							    size_t conv_v_id){
      size_t flat_index_up_to_corr = (facet_id * params.cube_channel_dim_size + grid_channel_id) * params.number_of_polarization_terms_being_gridded;
      size_t reduction_step_size = convolution_full_support * convolution_full_support;
      size_t current_reduction_step_flat_index = conv_v_id * convolution_full_support + conv_u_id;
      atomicAdd(params.normalization_terms + (flat_index_up_to_corr * reduction_step_size + current_reduction_step_flat_index),accumulator._x);
      atomicAdd(params.normalization_terms + ((flat_index_up_to_corr + 1) * reduction_step_size + current_reduction_step_flat_index), accumulator._y);
      atomicAdd(params.normalization_terms + ((flat_index_up_to_corr + 2) * reduction_step_size + current_reduction_step_flat_index), accumulator._z);
      atomicAdd(params.normalization_terms + ((flat_index_up_to_corr + 3) * reduction_step_size + current_reduction_step_flat_index), accumulator._w);
    }
  };
  template <>
  class correlation_gridding_policy<grid_4_correlation_with_jones_corrections> {
  private:
    __device__ static size_t find_jones_index(size_t * antenna_jones_matricies,size_t count, size_t match_id){
	for (size_t i = 0; i < count; ++i){
	  size_t time_index_of_i = antenna_jones_matricies[i];
	  if (time_index_of_i == match_id)
	    return i;
	}
	return 2 << 31;
    }
  public:
    typedef correlation_gridding_traits<grid_4_correlation_with_jones_corrections> active_trait;
    __device__ static void read_corralation_data (gridding_parameters & params,
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
    __device__ static void read_channel_grid_index(const gridding_parameters & params,
						   size_t spw_channel_flat_index,
						   size_t & out){
	imaging::correlation_gridding_policy<grid_4_correlation>::read_channel_grid_index(params,spw_channel_flat_index,out);
    }
    __device__ static void read_and_apply_antenna_jones_terms(const gridding_parameters & params,
							      size_t row_index,
							      typename active_trait::vis_type & vis){
	size_t correlator_timestamp_id = params.timestamp_ids[row_index]; //this may / may not correspond to the time loop (depending on how many timestamps are missing per baseline
	size_t antenna_1_id = params.antenna_1_ids[row_index];
	size_t antenna_2_id = params.antenna_2_ids[row_index]; //necessary to make sure we apply the matricies in the right order
	/* Since we've had to compact the Jones matrix arrays to dimensions: #antenna . time(antenna_id) . #spw . #channel . 4 correlations
	 * and the second term varies per antenna we need an array to check where each antenna's jones matricies start, as well
	 * as an array with the correlator timestamp index for every set (spw . channel . 4) of jones matricies on a per antenna basis
	 * We can now find the correct jones term based on its correlator timestamp index (remember some timestamps may be missing)
	 */
	size_t starting_jones_index_antenna_1 = params.antenna_jones_starting_indexes[antenna_1_id];
	size_t starting_jones_index_antenna_2 = params.antenna_jones_starting_indexes[antenna_2_id];
	size_t number_of_jones_terms_for_antenna_1 = params.antenna_jones_starting_indexes[antenna_1_id + 1] - 
						     starting_jones_index_antenna_1; //there are N+1 terms in this prefix scan
	size_t number_of_jones_terms_for_antenna_2 = params.antenna_jones_starting_indexes[antenna_2_id + 1] - 
						     starting_jones_index_antenna_2; //there are N+1 terms in this prefix scan
	size_t jones_collection_size = params.antenna_count * params.spw_count * params.channel_count;
	
	size_t jones_term_index_id_antenna_1 = find_jones_index(params.jones_time_indicies_per_antenna + starting_jones_index_antenna_1,
								number_of_jones_terms_for_antenna_1,correlator_timestamp_id);
	size_t jones_term_index_id_antenna_2 = find_jones_index(params.jones_time_indicies_per_antenna + starting_jones_index_antenna_2,
								number_of_jones_terms_for_antenna_2,correlator_timestamp_id);
	jones_2x2<visibility_base_type> * antenna_1_jones_matricies = (jones_2x2<visibility_base_type> *) params.jones_terms + 
								      (starting_jones_index_antenna_1 * jones_collection_size);
	jones_2x2<visibility_base_type> * antenna_2_jones_matricies = (jones_2x2<visibility_base_type> *) params.jones_terms + 
								      (starting_jones_index_antenna_2 * jones_collection_size);
	vis = (antenna_1_jones_matricies[jones_term_index_id_antenna_1] * (vis * antenna_2_jones_matricies[jones_term_index_id_antenna_2])); //remember matricies don't commute!
    }
    __device__ static void compute_facet_grid_ptr(const gridding_parameters & params,
						  size_t facet_id,
						  size_t grid_size_in_floats,
						  grid_base_type ** facet_grid_starting_ptr){
	imaging::correlation_gridding_policy<grid_4_correlation>::compute_facet_grid_ptr(params,facet_id,grid_size_in_floats,facet_grid_starting_ptr);
    }
    __device__ static void grid_visibility (grid_base_type* grid,
					    size_t slice_size,
					    size_t nx,
					    size_t grid_channel_id,
					    size_t no_polarizations_being_gridded,
					    size_t pos_u,
					    size_t pos_v,
					    typename active_trait::accumulator_type & accumulator
					   ){
	imaging::correlation_gridding_policy<grid_4_correlation>::grid_visibility(grid,slice_size,nx,
										  grid_channel_id,
										  no_polarizations_being_gridded,
										  pos_u,pos_v,accumulator);
    }
    __device__ static void update_normalization_accumulator(const gridding_parameters & params,
							    typename active_trait::normalization_accumulator_type & accumulator,
							    size_t facet_id,
							    size_t grid_channel_id,
							    size_t convolution_full_support,
							    size_t conv_u_id,
							    size_t conv_v_id){
	imaging::correlation_gridding_policy<grid_4_correlation>::update_normalization_accumulator(params,accumulator,facet_id,grid_channel_id,
												   convolution_full_support,conv_u_id,conv_v_id);
    }
  };
}