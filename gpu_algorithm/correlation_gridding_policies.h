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
    __device__ static void read_corralation_data (gridding_parameters & params,
						  size_t row_index,
						  size_t spw,
						  size_t c,
						  typename active_trait::vis_type & vis,
						  typename active_trait::vis_flag_type & flag,
						  typename active_trait::vis_weight_type & weight
						 );
    __device__ static void grid_visibility (grid_base_type* grid,
					    size_t slice_size,
					    size_t nx,
					    size_t grid_channel_id,
					    size_t no_polarizations_being_gridded,
					    size_t pos_u,
					    size_t pos_v,
					    typename active_trait::accumulator_type & accumulator
					   );
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
  };
};