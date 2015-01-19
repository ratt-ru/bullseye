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
		size_t padded_conv_full_support = conv_full_support + 2; //remember we need to reserve some of the support for +/- frac on both sides
		size_t filter_size = padded_conv_full_support * params.conv_oversample;
		uvw_base_type conv_offset = (padded_conv_full_support) / 2.0; 
		for (size_t spw = 0; spw < params.spw_count; ++spw){
		    for (size_t c = 0; c < params.channel_count; ++c){ //best we can do is unroll and spill some registers... todo
			basic_complex<grid_base_type> my_grid_accum = {0,0};
			size_t my_previous_u = 0;
			size_t my_previous_v = 0;
			//read all the stuff that is only dependent on the current spw and channel
			size_t flat_indexed_spw_channel = spw * params.channel_count + c;
			bool channel_enabled = params.enabled_channels[flat_indexed_spw_channel];
			size_t channel_grid_index = params.channel_grid_indicies[flat_indexed_spw_channel];
			reference_wavelengths_base_type ref_wavelength = params.reference_wavelengths[flat_indexed_spw_channel];
			for (size_t t = 0; t < baseline_num_timestamps; ++t){				
				//read all the data we need for gridding
				size_t row = starting_row_index + t;
				size_t spw_index = params.spw_index_array[row];
				size_t vis_index = row * params.channel_count + c;
				bool currently_considering_spw = (spw == spw_index);
				imaging::uvw_coord<uvw_base_type> uvw = params.uvw_coords[row];
				bool row_flagged = params.flagged_rows[row];
				bool visibility_flagged = params.flags[vis_index];
				bool row_is_in_field_being_imaged = (params.field_array[row] == params.imaging_field);
				basic_complex<visibility_base_type> vis = ((basic_complex<visibility_base_type>*)params.visibilities)[vis_index];
				visibility_weights_base_type vis_weight = params.visibility_weights[vis_index];
				//compute the weighted visibility and promote the flags to integers so that we don't have unnecessary branch diversion here
				visibility_base_type combined_vis_weight = vis_weight * (visibility_base_type)(int)(!(row_flagged || visibility_flagged) && 
														    currently_considering_spw && 
														    channel_enabled && 
														    row_is_in_field_being_imaged);
				//scale the uv coordinates to the correct FOV by the fourier simularity theorem (pg 146-148 Synthesis Imaging in Radio Astronomy II)
				uvw._u *= u_scale;
				uvw._v *= v_scale;
				//measure the baseline in terms of wavelength
				uvw._u /= ref_wavelength;
				uvw._v /= ref_wavelength;
				//account for interpolation error (we select the closest sample from the oversampled convolution filter)
				uvw_base_type cont_current_u = uvw._u + grid_centre_offset_x - conv_offset;
				uvw_base_type cont_current_v = uvw._v + grid_centre_offset_y - conv_offset;
				size_t my_current_u = round(cont_current_u);
				size_t my_current_v = round(cont_current_v);
				size_t frac_u = (-cont_current_u + (uvw_base_type)my_current_u) * params.conv_oversample;
				size_t frac_v = (-cont_current_v + (uvw_base_type)my_current_v) * params.conv_oversample;
				//map the convolution memory access to a coalesced access (bundle #full_support number of fractions together, so that the memory addresses are contigious)
				size_t closest_conv_u = frac_u * padded_conv_full_support + my_conv_u;
				size_t closest_conv_v = frac_v * padded_conv_full_support + my_conv_v;
				
				my_current_u += my_conv_u;
				my_current_v += my_conv_v;
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
						grid_base_type* grid_flat_ptr = (grid_base_type*)params.output_buffer + 
										channel_grid_index * params.nx * params.ny + 
										(my_previous_v * params.nx + my_previous_u) * 2;
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
				my_grid_accum._real += vis._real * conv_weight * combined_vis_weight;
				my_grid_accum._imag += vis._imag * conv_weight * combined_vis_weight; 
			}
			//Okay this channel is done... now lets dump whatever has been accumulated since the last dump
			if (my_previous_u + conv_full_support  < params.ny && my_previous_u + conv_full_support  < params.nx &&
			    my_previous_v < params.ny && my_previous_u < params.nx){
				grid_base_type* grid_flat_ptr = (grid_base_type*)params.output_buffer + 
								channel_grid_index * params.nx * params.ny + 
								(my_previous_v * params.nx + my_previous_u) * 2;
				atomicAdd(grid_flat_ptr,my_grid_accum._real);
				atomicAdd(grid_flat_ptr + 1,my_grid_accum._imag);
			}
		    }
		}
	}
}
