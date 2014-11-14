#pragma once
#include "cu_common.h"
#include "gridding_parameters.h"

namespace imaging {
	/*
		This is a gridding kernel following Romeins distribution stategy.
		This should be launched with 
			block dimensions: {conv_dim_size,conv_dim_size,no_facets}
			blocks per grid: {no_baselines,1,1}
	 */
	__global__ void grid_single(gridding_parameters params,
                  dim3 no_blocks_per_grid, dim3 no_threads_per_block){
		size_t my_baseline = blockIdx.x;
		size_t my_conv_u = threadIdx.x;
		size_t my_conv_v = threadIdx.y;
		size_t starting_row_index = params.baseline_starting_indexes[my_baseline];
		//the starting index prescan must be n(n-1)/2 + n + 1 elements long since we need the length of the last baseline
		size_t baseline_num_timestamps = params.baseline_starting_indexes[my_baseline+1] - starting_row_index;
		uvw_base_type grid_centre_offset_x = params.nx/2.0;
		uvw_base_type grid_centre_offset_y = params.ny/2.0;
		uvw_base_type u_scale=params.nx*params.cell_size_x * ARCSEC_TO_RAD;
                uvw_base_type v_scale=params.ny*params.cell_size_y * ARCSEC_TO_RAD;

		uvw_base_type conv_offset = params.conv_support * params.conv_oversample / (uvw_base_type)2;

		for (size_t c = 0; c < params.channel_count; ++c){ //best we can do is unroll and spill some registers... todo
			basic_complex<grid_base_type> my_grid_accum = {0,0};
			size_t my_previous_u = 0;
			size_t my_previous_v = 0;
			for (size_t t = 0; t < baseline_num_timestamps; ++t){
				size_t row = starting_row_index + t;
				//read uvw
				imaging::uvw_coord<uvw_base_type> uvw = params.uvw_coords[row];
				uvw._u *= u_scale;
				uvw._v *= v_scale;
				size_t spw_index = params.spw_index_array[row];
				reference_wavelengths_base_type ref_wavelength = params.reference_wavelengths[spw_index * params.spw_count + c];
                                uvw._u /= ref_wavelength;
				uvw._v /= ref_wavelength;
				size_t my_current_u = round(uvw._u + (my_conv_u - conv_offset)/params.conv_oversample + grid_centre_offset_x);
				size_t my_current_v = round(uvw._v + (my_conv_v - conv_offset)/params.conv_oversample + grid_centre_offset_y);
				basic_complex<visibility_base_type> vis = ((basic_complex<visibility_base_type>*)params.visibilities)[row*params.channel_count + c];
				//todo:read row flag
				//todo:read channel flag
				//todo:read field id (should match field being imaged)
				//todo:read channel flags
				//if this is the first timestamp for this baseline initialize previous_u and previous_v
				if (t == 0) {
					my_previous_u = my_current_u;
					my_previous_v = my_current_v;
				}
				//if u and v have changed we must dump everything to memory at previous_u and previous_v and reset
				if (!(my_current_u == my_previous_u && my_current_v == my_previous_v)){
					if (my_previous_u < params.nx && my_previous_v < params.ny){
						size_t grid_flat_index = (my_previous_v * params.ny + params.nx) << 1;
						atomicAdd((grid_base_type*)params.output_buffer + grid_flat_index,my_grid_accum._real);
						atomicAdd((grid_base_type*)params.output_buffer + grid_flat_index + 1,my_grid_accum._imag);
					}
				}
				//todo:for now lets read the convolution weight... then we should try calculate it on the fly later on
				size_t conv_flat_index = my_conv_v * (params.conv_oversample * params.conv_support) + my_conv_u;
				convolution_base_type conv_weight = params.conv[conv_flat_index];	
				//then multiply-add into the accumulator 
				my_grid_accum._real += vis._real * conv_weight;
				my_grid_accum._imag += vis._imag * conv_weight; 
			}
		}
	}
}
