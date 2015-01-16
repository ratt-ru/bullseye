#pragma once
#include "cu_common.h"
#include "gridding_parameters.h"

namespace imaging {
	/*
		This is a gridding kernel following Romeins distribution stategy.
		This should be launched with 
			block dimensions: {conv_dim_size,conv_dim_size,1}
			blocks per grid: {no_baselines,1,1}
	 */
	__global__ void grid_single(gridding_parameters params,
                  dim3 no_blocks_per_grid, dim3 no_threads_per_block){
		size_t my_baseline = blockIdx.x;
		size_t my_conv_u = threadIdx.x + 1;
		size_t my_conv_v = threadIdx.y + 1;
		
		size_t starting_row_index = params.baseline_starting_indexes[my_baseline];
		//the starting index prescan must be n(n-1)/2 + n + 1 elements long since we need the length of the last baseline
		size_t baseline_num_timestamps = params.baseline_starting_indexes[my_baseline+1] - starting_row_index;
		
		//Scale the IFFT by the simularity theorem to the correct FOV
		uvw_base_type grid_centre_offset_x = params.nx/2.0;
		uvw_base_type grid_centre_offset_y = params.ny/2.0;
		uvw_base_type u_scale=params.nx*params.cell_size_x * ARCSEC_TO_RAD;
                uvw_base_type v_scale=-(params.ny*params.cell_size_y * ARCSEC_TO_RAD);
		
		size_t conv_full_support = params.conv_support * 2 + 1;
		uvw_base_type conv_offset = (conv_full_support + 2) / 2.0; //remember we need to reserve some of the support for +/- frac on both sides
		
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
				//measure the baseline in terms of wavelength
				size_t spw_index = params.spw_index_array[row];
				reference_wavelengths_base_type ref_wavelength = params.reference_wavelengths[spw_index * params.spw_count + c];
                                uvw._u /= ref_wavelength;
				uvw._v /= ref_wavelength;
				//account for interpolation error (we select the closest sample from the oversampled convolution filter)
				uvw_base_type cont_current_u = uvw._u + grid_centre_offset_x - conv_offset;
				uvw_base_type cont_current_v = uvw._v + grid_centre_offset_y - conv_offset;
				size_t my_current_u = round(cont_current_u);
				size_t my_current_v = round(cont_current_v);
				uvw_base_type frac_u = -cont_current_u + (uvw_base_type)my_current_u;
				uvw_base_type frac_v = -cont_current_v + (uvw_base_type)my_current_v;
				size_t closest_conv_u = ((uvw_base_type)my_conv_u + frac_u)*params.conv_oversample;
				size_t closest_conv_v = ((uvw_base_type)my_conv_v + frac_v)*params.conv_oversample;
				my_current_u += my_conv_u;
				my_current_v += my_conv_v;
				basic_complex<visibility_base_type> vis = ((basic_complex<visibility_base_type>*)params.visibilities)[row*params.channel_count + c];
				//todo:read row flag
				//todo:read channel flags
				//todo:read field id (should match field being imaged)
				//todo:read channel cube indexes
				//if this is the first timestamp for this baseline initialize previous_u and previous_v
				if (t == 0) {
					my_previous_u = my_current_u;
					my_previous_v = my_current_v;
				}
				//if u and v have changed we must dump everything to memory at previous_u and previous_v and reset
				if ((my_current_u != my_previous_u || my_current_v != my_previous_v)){
					//don't you dare go off the grid:
					if (my_previous_v + conv_full_support  < params.ny && my_previous_u + conv_full_support  < params.nx &&
					    my_previous_v < params.ny && my_previous_u < params.nx){
						grid_base_type* grid_flat_ptr = (grid_base_type*)params.output_buffer + (my_previous_v * params.nx + my_previous_u) * 2;
						atomicAdd(grid_flat_ptr,my_grid_accum._real);
						atomicAdd(grid_flat_ptr + 1,my_grid_accum._imag);
					}
					my_grid_accum._real = 0;
					my_grid_accum._imag = 0;
					my_previous_u = my_current_u;
					my_previous_v = my_current_v;
				}
				//Lets read the convolution weights from the the precomputed filter
				convolution_base_type conv_weight = params.conv[closest_conv_u] * params.conv[closest_conv_v];
				
				//then multiply-add into the accumulator 				
				my_grid_accum._real += vis._real * conv_weight;
				my_grid_accum._imag += vis._imag * conv_weight; 
			}
			//Okay this channel is done... now lets dump whatever has been accumulated since the last dump
			if (my_previous_u + conv_full_support  < params.ny && my_previous_u + conv_full_support  < params.nx &&
			    my_previous_v < params.ny && my_previous_u < params.nx){
				grid_base_type* grid_flat_ptr = (grid_base_type*)params.output_buffer + (my_previous_v * params.nx + my_previous_u) * 2;
				atomicAdd(grid_flat_ptr,my_grid_accum._real);
				atomicAdd(grid_flat_ptr + 1,my_grid_accum._imag);
			}
		}
	}
}
