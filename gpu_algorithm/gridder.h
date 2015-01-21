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
		uvw_base_type u_scale=params.nx*params.cell_size_x * ARCSEC_TO_RAD;
                uvw_base_type v_scale=-(params.ny*params.cell_size_y * ARCSEC_TO_RAD);
		
		size_t conv_full_support = params.conv_support * 2 + 1;
		size_t padded_conv_full_support = conv_full_support + 2; //remember we need to reserve some of the support for +/- frac on both sides
		uvw_base_type conv_offset = (padded_conv_full_support) / 2.0; 
		
		uvw_base_type grid_centre_offset_x = params.nx/2.0 - conv_offset + my_conv_u;
		uvw_base_type grid_centre_offset_y = params.ny/2.0 - conv_offset + my_conv_v;
		
		//load the convolution filter into shared memory
		extern __shared__ convolution_base_type shared_conv[];
		if (threadIdx.x == 0){
		  for (size_t x = 0; x < params.conv_oversample * padded_conv_full_support; ++x){
		    shared_conv[x] = params.conv[x];
		  }
		}
		__syncthreads(); //wait for the first thread to put the entire filter into shared memory
		
		size_t channel_loop_ubound = params.channel_count/2;
		size_t channel_loop_rem_ubound = channel_loop_ubound + params.channel_count % 2;
		//we must keep seperate accumulators per spw and channel, so we need to bring these loops outward (contrary to Romein's paper)
		for (size_t spw = 0; spw < params.spw_count; ++spw){
		    for (size_t c = 0; c < channel_loop_ubound; c += 2){
			basic_complex<grid_base_type> my_grid_accum = {0,0};
			basic_complex<grid_base_type> my_grid_accum_2 = {0,0};
// 			basic_complex<grid_base_type> my_grid_accum_3 = {0,0};
// 			basic_complex<grid_base_type> my_grid_accum_4 = {0,0};
			size_t my_previous_u = 0;
			size_t my_previous_u_2 = 0;
// 			size_t my_previous_u_3 = 0;
// 			size_t my_previous_u_4 = 0;
			size_t my_previous_v = 0;
			size_t my_previous_v_2 = 0;
// 			size_t my_previous_v_3 = 0;
// 			size_t my_previous_v_4 = 0;
			//read all the stuff that is only dependent on the current spw and channel
			size_t flat_indexed_spw_channel = spw * params.channel_count + c;
			size_t flat_indexed_spw_channel_2 = flat_indexed_spw_channel + 1;
// 			size_t flat_indexed_spw_channel_3 = flat_indexed_spw_channel + 2;
// 			size_t flat_indexed_spw_channel_4 = flat_indexed_spw_channel + 3;
			bool channel_enabled = params.enabled_channels[flat_indexed_spw_channel];
			bool channel_enabled_2 = params.enabled_channels[flat_indexed_spw_channel_2];
// 			bool channel_enabled_3 = params.enabled_channels[flat_indexed_spw_channel_3];
// 			bool channel_enabled_4 = params.enabled_channels[flat_indexed_spw_channel_4];
			size_t channel_grid_index = params.channel_grid_indicies[flat_indexed_spw_channel];
			size_t channel_grid_index_2 = params.channel_grid_indicies[flat_indexed_spw_channel_2];
// 			size_t channel_grid_index_3 = params.channel_grid_indicies[flat_indexed_spw_channel_3];
// 			size_t channel_grid_index_4 = params.channel_grid_indicies[flat_indexed_spw_channel_4];
			reference_wavelengths_base_type ref_wavelength = 1 / params.reference_wavelengths[flat_indexed_spw_channel];
			reference_wavelengths_base_type ref_wavelength_2 = 1 / params.reference_wavelengths[flat_indexed_spw_channel_2];
// 			reference_wavelengths_base_type ref_wavelength_3 = 1 / params.reference_wavelengths[flat_indexed_spw_channel_3];
// 			reference_wavelengths_base_type ref_wavelength_4 = 1 / params.reference_wavelengths[flat_indexed_spw_channel_4];
			for (size_t t = 0; t < baseline_num_timestamps; ++t){
				//read all the data we need for gridding
				size_t row = starting_row_index + t;
				size_t spw_index = params.spw_index_array[row];
				size_t vis_index = row * params.channel_count + c;
				size_t vis_index_2 = vis_index+1;
// 				size_t vis_index_3 = vis_index+2;
// 				size_t vis_index_4 = vis_index+3;
				bool currently_considering_spw = (spw == spw_index);
				imaging::uvw_coord<uvw_base_type> uvw = params.uvw_coords[row];
				imaging::uvw_coord<uvw_base_type> uvw_2 = uvw;
// 				imaging::uvw_coord<uvw_base_type> uvw_3 = uvw;
// 				imaging::uvw_coord<uvw_base_type> uvw_4 = uvw;
				bool row_flagged = params.flagged_rows[row];
				bool visibility_flagged = params.flags[vis_index];
				bool visibility_flagged_2 = params.flags[vis_index_2];
// 				bool visibility_flagged_3 = params.flags[vis_index_2];
// 				bool visibility_flagged_4 = params.flags[vis_index_2];
				bool row_is_in_field_being_imaged = (params.field_array[row] == params.imaging_field);
				basic_complex<visibility_base_type> vis = ((basic_complex<visibility_base_type>*)params.visibilities)[vis_index];
				basic_complex<visibility_base_type> vis_2 = ((basic_complex<visibility_base_type>*)params.visibilities)[vis_index_2];
// 				basic_complex<visibility_base_type> vis_3 = ((basic_complex<visibility_base_type>*)params.visibilities)[vis_index_3];
// 				basic_complex<visibility_base_type> vis_4 = ((basic_complex<visibility_base_type>*)params.visibilities)[vis_index_4];
				visibility_weights_base_type vis_weight = params.visibility_weights[vis_index];
				visibility_weights_base_type vis_weight_2 = params.visibility_weights[vis_index_2];
// 				visibility_weights_base_type vis_weight_3 = params.visibility_weights[vis_index_3];
// 				visibility_weights_base_type vis_weight_4 = params.visibility_weights[vis_index_4];
				//compute the weighted visibility and promote the flags to integers so that we don't have unnecessary branch diversion here
				visibility_base_type combined_vis_weight = vis_weight * (visibility_base_type)(int)(!(row_flagged || visibility_flagged) && 
														    currently_considering_spw && 
														    channel_enabled && 
														    row_is_in_field_being_imaged);
				visibility_base_type combined_vis_weight_2 = vis_weight_2 * (visibility_base_type)(int)(!(row_flagged || visibility_flagged_2) && 
														    currently_considering_spw && 
														    channel_enabled_2 && 
														    row_is_in_field_being_imaged);
// 				visibility_base_type combined_vis_weight_3 = vis_weight_3 * (visibility_base_type)(int)(!(row_flagged || visibility_flagged_3) && 
// 														    currently_considering_spw && 
// 														    channel_enabled_3 && 
// 														    row_is_in_field_being_imaged);
// 				visibility_base_type combined_vis_weight_4 = vis_weight_4 * (visibility_base_type)(int)(!(row_flagged || visibility_flagged_4) && 
// 														    currently_considering_spw && 
// 														    channel_enabled_4 && 
// 														    row_is_in_field_being_imaged);
				//scale the uv coordinates (measured in wavelengths) to the correct FOV by the fourier simularity theorem (pg 146-148 Synthesis Imaging in Radio Astronomy II)
				uvw._u *= u_scale * ref_wavelength; 
				uvw._v *= v_scale * ref_wavelength;
				uvw_2._u *= u_scale * ref_wavelength_2; 
				uvw_2._v *= v_scale * ref_wavelength_2;
// 				uvw_3._u *= u_scale * ref_wavelength_3; 
// 				uvw_3._v *= v_scale * ref_wavelength_3;
// 				uvw_4._u *= u_scale * ref_wavelength_4; 
// 				uvw_4._v *= v_scale * ref_wavelength_4;
				//account for interpolation error (we select the closest sample from the oversampled convolution filter)
				uvw_base_type cont_current_u = uvw._u + grid_centre_offset_x;
				uvw_base_type cont_current_v = uvw._v + grid_centre_offset_y;
				uvw_base_type cont_current_u_2 = uvw_2._u + grid_centre_offset_x;
				uvw_base_type cont_current_v_2 = uvw_2._v + grid_centre_offset_y;
// 				uvw_base_type cont_current_u_3 = uvw_3._u + grid_centre_offset_x;
// 				uvw_base_type cont_current_v_3 = uvw_3._v + grid_centre_offset_y;
// 				uvw_base_type cont_current_u_4 = uvw_4._u + grid_centre_offset_x;
// 				uvw_base_type cont_current_v_4 = uvw_4._v + grid_centre_offset_y;
				size_t my_current_u = round(cont_current_u);
				size_t my_current_v = round(cont_current_v);
				size_t my_current_u_2 = round(cont_current_u_2);
				size_t my_current_v_2 = round(cont_current_v_2);
// 				size_t my_current_u_3 = round(cont_current_u_3);
// 				size_t my_current_v_3 = round(cont_current_v_3);
// 				size_t my_current_u_4 = round(cont_current_u_4);
// 				size_t my_current_v_4 = round(cont_current_v_4);
				uvw_base_type frac_u = (-cont_current_u + (uvw_base_type)my_current_u);
				uvw_base_type frac_v = (-cont_current_v + (uvw_base_type)my_current_v);
				uvw_base_type frac_u_2 = (-cont_current_u_2 + (uvw_base_type)my_current_u_2);
				uvw_base_type frac_v_2 = (-cont_current_v_2 + (uvw_base_type)my_current_v_2);
// 				uvw_base_type frac_u_3 = (-cont_current_u_3 + (uvw_base_type)my_current_u_3);
// 				uvw_base_type frac_v_3 = (-cont_current_v_3 + (uvw_base_type)my_current_v_3);
// 				uvw_base_type frac_u_4 = (-cont_current_u_4 + (uvw_base_type)my_current_u_4);
// 				uvw_base_type frac_v_4 = (-cont_current_v_4 + (uvw_base_type)my_current_v_4);
				//map the convolution memory access to a coalesced access (bundle #full_support number of fractions together, so that the memory addresses are contigious)
				size_t closest_conv_u = (frac_u + (uvw_base_type)my_conv_u)* params.conv_oversample;
				size_t closest_conv_v = (frac_v + (uvw_base_type)my_conv_v)* params.conv_oversample;
				size_t closest_conv_u_2 = (frac_u_2 + (uvw_base_type)my_conv_u)* params.conv_oversample;
				size_t closest_conv_v_2 = (frac_v_2 + (uvw_base_type)my_conv_v)* params.conv_oversample;
// 				size_t closest_conv_u_3 = (frac_u_3 + (uvw_base_type)my_conv_u)* params.conv_oversample;
// 				size_t closest_conv_v_3 = (frac_v_3 + (uvw_base_type)my_conv_v)* params.conv_oversample;
// 				size_t closest_conv_u_4 = (frac_u_4 + (uvw_base_type)my_conv_u)* params.conv_oversample;
// 				size_t closest_conv_v_4 = (frac_v_4 + (uvw_base_type)my_conv_v)* params.conv_oversample;
				//if this is the first timestamp for this baseline initialize previous_u and previous_v
				if (t == 0) {
					my_previous_u = my_current_u;
					my_previous_v = my_current_v;
					my_previous_u_2 = my_current_u_2;
					my_previous_v_2 = my_current_v_2;
// 					my_previous_u_3 = my_current_u_3;
// 					my_previous_v_3 = my_current_v_3;
// 					my_previous_u_4 = my_current_u_4;
// 					my_previous_v_4 = my_current_v_4;
				}
				//if u and v have changed we must dump everything to memory at previous_u and previous_v and reset
				if ((my_current_u != my_previous_u || my_current_v != my_previous_v)){
					//don't you dare go off the grid:
					if (my_previous_v + conv_full_support  < params.ny && my_previous_u + conv_full_support  < params.nx &&
					    my_previous_v < params.ny && my_previous_u < params.nx){
						grid_base_type* grid_flat_ptr = (grid_base_type*)params.output_buffer + 
										channel_grid_index * params.nx * params.ny + 
										((my_previous_v * params.nx + my_previous_u) << 1);
						atomicAdd(grid_flat_ptr,my_grid_accum._real);
						atomicAdd(grid_flat_ptr + 1,my_grid_accum._imag);
					}
					my_grid_accum._real = 0;
					my_grid_accum._imag = 0;
					my_previous_u = my_current_u;
					my_previous_v = my_current_v;
				}
				//if u and v have changed we must dump everything to memory at previous_u and previous_v and reset
				if ((my_current_u_2 != my_previous_u_2 || my_current_v_2 != my_previous_v_2)){
					//don't you dare go off the grid:
					if (my_previous_v_2 + conv_full_support  < params.ny && my_previous_u_2 + conv_full_support  < params.nx &&
					    my_previous_v_2 < params.ny && my_previous_u_2 < params.nx){
						grid_base_type* grid_flat_ptr = (grid_base_type*)params.output_buffer + 
										channel_grid_index_2 * params.nx * params.ny + 
										((my_previous_v_2 * params.nx + my_previous_u_2) << 1);
						atomicAdd(grid_flat_ptr,my_grid_accum_2._real);
						atomicAdd(grid_flat_ptr + 1,my_grid_accum_2._imag);
					}
					my_grid_accum_2._real = 0;
					my_grid_accum_2._imag = 0;
					my_previous_u_2 = my_current_u_2;
					my_previous_v_2 = my_current_v_2;
				}
				//if u and v have changed we must dump everything to memory at previous_u and previous_v and reset
// 				if ((my_current_u_3 != my_previous_u_3 || my_current_v_3 != my_previous_v_3)){
// 					//don't you dare go off the grid:
// 					if (my_previous_v_3 + conv_full_support  < params.ny && my_previous_u_3 + conv_full_support  < params.nx &&
// 					    my_previous_v_3 < params.ny && my_previous_u_3 < params.nx){
// 						grid_base_type* grid_flat_ptr = (grid_base_type*)params.output_buffer + 
// 										channel_grid_index_3 * params.nx * params.ny + 
// 										((my_previous_v_3 * params.nx + my_previous_u_3) << 1);
// 						atomicAdd(grid_flat_ptr,my_grid_accum_3._real);
// 						atomicAdd(grid_flat_ptr + 1,my_grid_accum_3._imag);
// 					}
// 					my_grid_accum_3._real = 0;
// 					my_grid_accum_3._imag = 0;
// 					my_previous_u_3 = my_current_u_3;
// 					my_previous_v_3 = my_current_v_3;
// 				}
				//if u and v have changed we must dump everything to memory at previous_u and previous_v and reset
// 				if ((my_current_u_4 != my_previous_u_4 || my_current_v_4 != my_previous_v_4)){
// 					//don't you dare go off the grid:
// 					if (my_previous_v_4 + conv_full_support  < params.ny && my_previous_u_4 + conv_full_support  < params.nx &&
// 					    my_previous_v_4 < params.ny && my_previous_u_4 < params.nx){
// 						grid_base_type* grid_flat_ptr = (grid_base_type*)params.output_buffer + 
// 										channel_grid_index_4 * params.nx * params.ny + 
// 										((my_previous_v_4 * params.nx + my_previous_u_4) << 1);
// 						atomicAdd(grid_flat_ptr,my_grid_accum_4._real);
// 						atomicAdd(grid_flat_ptr + 1,my_grid_accum_4._imag);
// 					}
// 					my_grid_accum_4._real = 0;
// 					my_grid_accum_4._imag = 0;
// 					my_previous_u_4 = my_current_u_4;
// 					my_previous_v_4 = my_current_v_4;
// 				}
				//Lets read the convolution weights from the the precomputed filter
				convolution_base_type conv_weight = shared_conv[closest_conv_u] * shared_conv[closest_conv_v];	
				convolution_base_type conv_weight_2 = shared_conv[closest_conv_u_2] * shared_conv[closest_conv_v_2];
// 				convolution_base_type conv_weight_3 = shared_conv[closest_conv_u_3] * shared_conv[closest_conv_v_3];
// 				convolution_base_type conv_weight_4 = shared_conv[closest_conv_u_4] * shared_conv[closest_conv_v_4];	
				//then multiply-add into the accumulator 				
				my_grid_accum._real += vis._real * conv_weight * combined_vis_weight;
				my_grid_accum._imag += vis._imag * conv_weight * combined_vis_weight;
				my_grid_accum_2._real += vis_2._real * conv_weight_2 * combined_vis_weight_2;
				my_grid_accum_2._imag += vis_2._imag * conv_weight_2 * combined_vis_weight_2; 
// 				my_grid_accum_3._real += vis_3._real * conv_weight_3 * combined_vis_weight_3;
// 				my_grid_accum_3._imag += vis_3._imag * conv_weight_3 * combined_vis_weight_3; 
// 				my_grid_accum_4._real += vis_4._real * conv_weight_4 * combined_vis_weight_4;
// 				my_grid_accum_4._imag += vis_4._imag * conv_weight_4 * combined_vis_weight_4; 
			}
			//Okay this channel is done... now lets dump whatever has been accumulated since the last dump
			if (my_previous_v + conv_full_support  < params.ny && my_previous_u + conv_full_support  < params.nx &&
			    my_previous_v < params.ny && my_previous_u < params.nx){
				grid_base_type* grid_flat_ptr = (grid_base_type*)params.output_buffer + 
								channel_grid_index * params.nx * params.ny + 
								((my_previous_v * params.nx + my_previous_u) << 1);
				atomicAdd(grid_flat_ptr,my_grid_accum._real);
				atomicAdd(grid_flat_ptr + 1,my_grid_accum._imag);
			}
			if (my_previous_v_2 + conv_full_support  < params.ny && my_previous_u_2 + conv_full_support  < params.nx &&
			    my_previous_v_2 < params.ny && my_previous_u_2 < params.nx){
				grid_base_type* grid_flat_ptr = (grid_base_type*)params.output_buffer + 
								channel_grid_index_2 * params.nx * params.ny + 
								((my_previous_v_2 * params.nx + my_previous_u_2) << 1);
				atomicAdd(grid_flat_ptr,my_grid_accum_2._real);
				atomicAdd(grid_flat_ptr + 1,my_grid_accum_2._imag);
			}
// 			if (my_previous_v_3 + conv_full_support  < params.ny && my_previous_u_3 + conv_full_support  < params.nx &&
// 			    my_previous_v_3 < params.ny && my_previous_u_3 < params.nx){
// 				grid_base_type* grid_flat_ptr = (grid_base_type*)params.output_buffer + 
// 								channel_grid_index_3 * params.nx * params.ny + 
// 								((my_previous_v_3 * params.nx + my_previous_u_3) << 1);
// 				atomicAdd(grid_flat_ptr,my_grid_accum_3._real);
// 				atomicAdd(grid_flat_ptr + 1,my_grid_accum_3._imag);
// 			}
// 			if (my_previous_v_4 + conv_full_support  < params.ny && my_previous_u_4 + conv_full_support  < params.nx &&
// 			    my_previous_v_4 < params.ny && my_previous_u_4 < params.nx){
// 				grid_base_type* grid_flat_ptr = (grid_base_type*)params.output_buffer + 
// 								channel_grid_index_4 * params.nx * params.ny + 
// 								((my_previous_v_4 * params.nx + my_previous_u_4) << 1);
// 				atomicAdd(grid_flat_ptr,my_grid_accum_4._real);
// 				atomicAdd(grid_flat_ptr + 1,my_grid_accum_4._imag);
// 			}
		    }
		    for (size_t c = channel_loop_ubound; c < channel_loop_rem_ubound; ++c){
			basic_complex<grid_base_type> my_grid_accum = {0,0};
			size_t my_previous_u = 0;
			size_t my_previous_v = 0;
			//read all the stuff that is only dependent on the current spw and channel
			size_t flat_indexed_spw_channel = spw * params.channel_count + c;
			bool channel_enabled = params.enabled_channels[flat_indexed_spw_channel];
			size_t channel_grid_index = params.channel_grid_indicies[flat_indexed_spw_channel];
			reference_wavelengths_base_type ref_wavelength = 1 / params.reference_wavelengths[flat_indexed_spw_channel];
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
				//scale the uv coordinates (measured in wavelengths) to the correct FOV by the fourier simularity theorem (pg 146-148 Synthesis Imaging in Radio Astronomy II)
				uvw._u *= u_scale * ref_wavelength; 
				uvw._v *= v_scale * ref_wavelength;
				//account for interpolation error (we select the closest sample from the oversampled convolution filter)
				uvw_base_type cont_current_u = uvw._u + grid_centre_offset_x;
				uvw_base_type cont_current_v = uvw._v + grid_centre_offset_y;
				size_t my_current_u = round(cont_current_u);
				size_t my_current_v = round(cont_current_v);
				uvw_base_type frac_u = (-cont_current_u + (uvw_base_type)my_current_u);
				uvw_base_type frac_v = (-cont_current_v + (uvw_base_type)my_current_v);
				//map the convolution memory access to a coalesced access (bundle #full_support number of fractions together, so that the memory addresses are contigious)
				size_t closest_conv_u = (frac_u + (uvw_base_type)my_conv_u)* params.conv_oversample;
				size_t closest_conv_v = (frac_v + (uvw_base_type)my_conv_v)* params.conv_oversample;
				
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
										((my_previous_v * params.nx + my_previous_u) << 1);
						atomicAdd(grid_flat_ptr,my_grid_accum._real);
						atomicAdd(grid_flat_ptr + 1,my_grid_accum._imag);
					}
					my_grid_accum._real = 0;
					my_grid_accum._imag = 0;
					my_previous_u = my_current_u;
					my_previous_v = my_current_v;
				}
				//Lets read the convolution weights from the the precomputed filter
				convolution_base_type conv_weight = shared_conv[closest_conv_u] * shared_conv[closest_conv_v];	
				//then multiply-add into the accumulator 				
				my_grid_accum._real += vis._real * conv_weight * combined_vis_weight;
				my_grid_accum._imag += vis._imag * conv_weight * combined_vis_weight; 
			}
			//Okay this channel is done... now lets dump whatever has been accumulated since the last dump
			if (my_previous_u + conv_full_support  < params.ny && my_previous_u + conv_full_support  < params.nx &&
			    my_previous_v < params.ny && my_previous_u < params.nx){
				grid_base_type* grid_flat_ptr = (grid_base_type*)params.output_buffer + 
								channel_grid_index * params.nx * params.ny + 
								((my_previous_v * params.nx + my_previous_u) << 1);
				atomicAdd(grid_flat_ptr,my_grid_accum._real);
				atomicAdd(grid_flat_ptr + 1,my_grid_accum._imag);
			}
		    }
		}
	}
}
