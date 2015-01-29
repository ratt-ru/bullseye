#pragma once
#include "cu_common.h"
#include "gridding_parameters.h"

namespace imaging {
	/*
		This is a gridding kernel following Romeins distribution stategy.
		This should be launched with 
			block dimensions: {THREADS_PER_BLOCK,1,1}
			blocks per grid: {minimum number of blocks required to run baselines*conv_support_size^^2 threads,
					  1,1}
	 */
	__global__ void grid_single(gridding_parameters params){
		size_t tid = cu_indexing_schemes::getGlobalIdx_1D_1D(gridDim,blockIdx,blockDim,threadIdx);
		size_t conv_full_support = (params.conv_support << 1) + 1;
		size_t conv_full_support_sq = conv_full_support * conv_full_support;
		size_t padded_conv_full_support = conv_full_support + 2; //remember we need to reserve some of the support for +/- frac on both sides
		if (tid >= params.baseline_count * conv_full_support_sq) return;
		size_t my_baseline = tid / conv_full_support_sq;
		size_t conv_theadid_flat_index = tid % conv_full_support_sq;
		size_t my_conv_v = (conv_theadid_flat_index / conv_full_support) + 1;
		size_t my_conv_u = (conv_theadid_flat_index % conv_full_support) + 1;
		
		size_t starting_row_index = params.baseline_starting_indexes[my_baseline];
		//the starting index prescan must be n(n-1)/2 + n + 1 elements long since we need the length of the last baseline
		size_t baseline_num_timestamps = params.baseline_starting_indexes[my_baseline+1] - starting_row_index;
		
		//Scale the IFFT by the simularity theorem to the correct FOV
		uvw_base_type u_scale=params.nx*params.cell_size_x * ARCSEC_TO_RAD;
		uvw_base_type v_scale=-(params.ny*params.cell_size_y * ARCSEC_TO_RAD);
		
		uvw_base_type conv_offset = (padded_conv_full_support) / 2.0; 
		uvw_base_type grid_centre_offset_x = params.nx/2.0 - conv_offset + my_conv_u;
		uvw_base_type grid_centre_offset_y = params.ny/2.0 - conv_offset + my_conv_v;
		size_t grid_size_in_floats = params.nx * params.ny << 1;
		
		//load the convolution filter into shared memory
		extern __shared__ convolution_base_type shared_conv[];
		if (threadIdx.x == 0){
		  size_t fir_ubound = ((params.conv_oversample * padded_conv_full_support));
		  
		  for (size_t x = 0; x < fir_ubound; ++x){
		    shared_conv[x] = params.conv[x];
		  }
		}
		__syncthreads(); //wait for the first thread to put the entire filter into shared memory
		
		size_t channel_loop_ubound = params.channel_count >> 1;
		size_t channel_loop_rem_lbound = channel_loop_ubound << 1;
		//we must keep seperate accumulators per spw and channel, so we need to bring these loops outward (contrary to Romein's paper)
		for (size_t spw = 0; spw < params.spw_count; ++spw){
		    for (size_t c_i = 0; c_i < channel_loop_ubound; ++c_i){
			size_t c = c_i << 1;
			basic_complex<grid_base_type> my_grid_accum = {0,0};
			basic_complex<grid_base_type> my_grid_accum_2 = {0,0};
			size_t my_previous_u = 0;
			size_t my_previous_u_2 = 0;
			size_t my_previous_v = 0;
			size_t my_previous_v_2 = 0;
			//read all the stuff that is only dependent on the current spw and channel
			size_t flat_indexed_spw_channel = spw * params.channel_count + c;
			size_t flat_indexed_spw_channel_2 = flat_indexed_spw_channel + 1;
			bool channel_enabled = params.enabled_channels[flat_indexed_spw_channel];
			bool channel_enabled_2 = params.enabled_channels[flat_indexed_spw_channel_2];
			size_t channel_grid_index = params.channel_grid_indicies[flat_indexed_spw_channel];
			size_t channel_grid_index_2 = params.channel_grid_indicies[flat_indexed_spw_channel_2];
			grid_base_type* grid_flat_ptr = (grid_base_type*)params.output_buffer + 
							((channel_grid_index * grid_size_in_floats));
			grid_base_type* grid_flat_ptr_2 = (grid_base_type*)params.output_buffer + 
							  ((channel_grid_index_2 * grid_size_in_floats));
			reference_wavelengths_base_type ref_wavelength = params.reference_wavelengths[flat_indexed_spw_channel];
			reference_wavelengths_base_type ref_wavelength_2 = params.reference_wavelengths[flat_indexed_spw_channel];
			ref_wavelength = 1/ref_wavelength;
			ref_wavelength_2 = 1/ref_wavelength_2;
			uvw_base_type u_scale_factor = u_scale * ref_wavelength;
			uvw_base_type v_scale_factor = v_scale * ref_wavelength;
			uvw_base_type u_scale_factor_2 = u_scale * ref_wavelength;
			uvw_base_type v_scale_factor_2 = v_scale * ref_wavelength;
			for (size_t t = 0; t < baseline_num_timestamps; ++t){
				//read all the data we need for gridding
				size_t row = starting_row_index + t;
				size_t spw_index = params.spw_index_array[row];
				size_t vis_index = row * params.channel_count + c;
				size_t vis_index_2 = vis_index+1;
				bool currently_considering_spw = (spw == spw_index);
				bool row_is_in_field_being_imaged = (params.field_array[row] == params.imaging_field);
				bool currently_considering_row = currently_considering_spw && row_is_in_field_being_imaged;
				imaging::uvw_coord<uvw_base_type> uvw = params.uvw_coords[row];
				imaging::uvw_coord<uvw_base_type> uvw_2 = uvw;
				bool row_flagged = params.flagged_rows[row];
				bool visibility_flagged = params.flags[vis_index];
				bool visibility_flagged_2 = params.flags[vis_index_2];
				
				basic_complex<visibility_base_type> vis = ((basic_complex<visibility_base_type>*)params.visibilities)[vis_index];
				basic_complex<visibility_base_type> vis_2 = ((basic_complex<visibility_base_type>*)params.visibilities)[vis_index_2];
				visibility_weights_base_type vis_weight = params.visibility_weights[vis_index];
				visibility_weights_base_type vis_weight_2 = params.visibility_weights[vis_index_2];
				//compute the weighted visibility and promote the flags to integers so that we don't have unnecessary branch diversion here
				visibility_base_type combined_vis_weight = vis_weight * (visibility_base_type)(int)(!(row_flagged || visibility_flagged) && 
														    channel_enabled && 
														    currently_considering_row);
				visibility_base_type combined_vis_weight_2 = vis_weight_2 * (visibility_base_type)(int)(!(row_flagged || visibility_flagged_2) && 
														    channel_enabled_2 && 
														    currently_considering_row);
				//scale the uv coordinates (measured in wavelengths) to the correct FOV by the fourier simularity theorem (pg 146-148 Synthesis Imaging in Radio Astronomy II)
				uvw._u *= u_scale_factor;
				uvw._v *= v_scale_factor;
				uvw_2._u *= u_scale_factor_2;
				uvw_2._v *= v_scale_factor_2;
				//account for interpolation error (we select the closest sample from the oversampled convolution filter)
				uvw_base_type cont_current_u = uvw._u + grid_centre_offset_x;
				uvw_base_type cont_current_v = uvw._v + grid_centre_offset_y;
				uvw_base_type cont_current_u_2 = uvw_2._u + grid_centre_offset_x;
				uvw_base_type cont_current_v_2 = uvw_2._v + grid_centre_offset_y;
				size_t my_current_u = rintf(cont_current_u);
				size_t my_current_v = rintf(cont_current_v);
				size_t my_current_u_2 = rintf(cont_current_u_2);
				size_t my_current_v_2 = rintf(cont_current_v_2);
				size_t frac_u = (-cont_current_u + (uvw_base_type)my_current_u) * params.conv_oversample;
				size_t frac_v = (-cont_current_v + (uvw_base_type)my_current_v) * params.conv_oversample;
				size_t frac_u_2 = (-cont_current_u_2 + (uvw_base_type)my_current_u_2) * params.conv_oversample;
				size_t frac_v_2 = (-cont_current_v_2 + (uvw_base_type)my_current_v_2) * params.conv_oversample;
				//map the convolution memory access to a coalesced access (bundle #full_support number of fractions together, so that the memory addresses are contigious)
				size_t closest_conv_u = frac_u * padded_conv_full_support + my_conv_u;
				size_t closest_conv_v = frac_v * padded_conv_full_support + my_conv_v;
				size_t closest_conv_u_2 = frac_u_2 * padded_conv_full_support + my_conv_u;
				size_t closest_conv_v_2 = frac_v_2 * padded_conv_full_support + my_conv_v;
				//if this is the first timestamp for this baseline initialize previous_u and previous_v
				if (t == 0) {
					my_previous_u = my_current_u;
					my_previous_v = my_current_v;
					my_previous_u_2 = my_current_u_2;
					my_previous_v_2 = my_current_v_2;
				}
				//if u and v have changed we must dump everything to memory at previous_u and previous_v and reset
				if (!(my_current_u == my_previous_u && my_current_v == my_previous_v) && channel_enabled){
					//don't you dare go off the grid:
					if (my_previous_v + conv_full_support  < params.ny && my_previous_u + conv_full_support  < params.nx &&
					    my_previous_v < params.ny && my_previous_u < params.nx){
						grid_base_type* grid_flat_index = grid_flat_ptr + ((my_previous_v * params.nx + my_previous_u) << 1);
						atomicAdd(grid_flat_index,my_grid_accum._real);
						atomicAdd(grid_flat_index + 1,my_grid_accum._imag);
					}
					my_grid_accum._real = 0;
					my_grid_accum._imag = 0;
					my_previous_u = my_current_u;
					my_previous_v = my_current_v;
				}
				//if u and v have changed we must dump everything to memory at previous_u and previous_v and reset
				if (!(my_current_u_2 == my_previous_u_2 && my_current_v_2 == my_previous_v_2) && channel_enabled_2){
					//don't you dare go off the grid:
					if (my_previous_v_2 + conv_full_support  < params.ny && my_previous_u_2 + conv_full_support  < params.nx &&
					    my_previous_v_2 < params.ny && my_previous_u_2 < params.nx){
						grid_base_type* grid_flat_index = grid_flat_ptr_2 + ((my_previous_v_2 * params.nx + my_previous_u_2) << 1);
						atomicAdd(grid_flat_index,my_grid_accum_2._real);
						atomicAdd(grid_flat_index + 1,my_grid_accum_2._imag);
					}
					my_grid_accum_2._real = 0;
					my_grid_accum_2._imag = 0;
					my_previous_u_2 = my_current_u_2;
					my_previous_v_2 = my_current_v_2;
				}
				//Lets read the convolution weights from the the precomputed filter
				convolution_base_type conv_weight = (shared_conv[closest_conv_u] * shared_conv[closest_conv_v]) * combined_vis_weight;
				convolution_base_type conv_weight_2 = (shared_conv[closest_conv_u_2] * shared_conv[closest_conv_v_2]) * combined_vis_weight_2;
				//then multiply-add into the accumulator 				
				my_grid_accum._real += vis._real * conv_weight;
				my_grid_accum._imag += vis._imag * conv_weight;
				my_grid_accum_2._real += vis_2._real * conv_weight_2;
				my_grid_accum_2._imag += vis_2._imag * conv_weight_2; 
			}
			//Okay this channel is done... now lets dump whatever has been accumulated since the last dump
			if (my_previous_v + conv_full_support  < params.ny && my_previous_u + conv_full_support  < params.nx &&
			      my_previous_v < params.ny && my_previous_u < params.nx && channel_enabled){
				grid_base_type* grid_flat_index = grid_flat_ptr + ((my_previous_v * params.nx + my_previous_u) << 1);
				atomicAdd(grid_flat_index,my_grid_accum._real);
				atomicAdd(grid_flat_index + 1,my_grid_accum._imag);
			}
			if (my_previous_v_2 + conv_full_support  < params.ny && my_previous_u_2 + conv_full_support  < params.nx &&
			      my_previous_v_2 < params.ny && my_previous_u_2 < params.nx && channel_enabled_2){
				grid_base_type* grid_flat_index = grid_flat_ptr_2 + ((my_previous_v_2 * params.nx + my_previous_u_2) << 1);
				atomicAdd(grid_flat_index,my_grid_accum_2._real);
				atomicAdd(grid_flat_index + 1,my_grid_accum_2._imag);
			}
		    }
		    for (size_t c = channel_loop_rem_lbound; c < params.channel_count; ++c){
			basic_complex<grid_base_type> my_grid_accum = {0,0};
			size_t my_previous_u = 0;
			size_t my_previous_v = 0;
			//read all the stuff that is only dependent on the current spw and channel
			size_t flat_indexed_spw_channel = spw * params.channel_count + c;
			bool channel_enabled = params.enabled_channels[flat_indexed_spw_channel];
			size_t channel_grid_index = params.channel_grid_indicies[flat_indexed_spw_channel];
			grid_base_type* grid_flat_ptr = (grid_base_type*)params.output_buffer + 
							((channel_grid_index * params.nx * params.ny) << 1);
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
				size_t my_current_u = rintf(cont_current_u);
				size_t my_current_v = rintf(cont_current_v);
				size_t frac_u = (-cont_current_u + (uvw_base_type)my_current_u) * params.conv_oversample;
				size_t frac_v = (-cont_current_v + (uvw_base_type)my_current_v) * params.conv_oversample;
				//map the convolution memory access to a coalesced access (bundle #full_support number of fractions together, so that the memory addresses are contigious)
				size_t closest_conv_u = frac_u * padded_conv_full_support + my_conv_u;
				size_t closest_conv_v = frac_v * padded_conv_full_support + my_conv_v;
				//if this is the first timestamp for this baseline initialize previous_u and previous_v
				if (t == 0) {
					my_previous_u = my_current_u;
					my_previous_v = my_current_v;
				}
				//if u and v have changed we must dump everything to memory at previous_u and previous_v and reset
				if ((my_current_u != my_previous_u || my_current_v != my_previous_v) && channel_enabled){
					//don't you dare go off the grid:
					if (my_previous_v + conv_full_support  < params.ny && my_previous_u + conv_full_support  < params.nx &&
					    my_previous_v < params.ny && my_previous_u < params.nx){
						grid_base_type* grid_flat_index = grid_flat_ptr + ((my_previous_v * params.nx + my_previous_u) << 1);
						atomicAdd(grid_flat_index,my_grid_accum._real);
						atomicAdd(grid_flat_index + 1,my_grid_accum._imag);
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
			if (channel_enabled)
			  if (my_previous_u + conv_full_support < params.ny && my_previous_u + conv_full_support  < params.nx &&
			      my_previous_v < params.ny && my_previous_u < params.nx){
				grid_base_type* grid_flat_ptr = (grid_base_type*)params.output_buffer + 
								((channel_grid_index * params.nx * params.ny  + 
								  my_previous_v * params.nx + my_previous_u) << 1);
				atomicAdd(grid_flat_ptr,my_grid_accum._real);
				atomicAdd(grid_flat_ptr + 1,my_grid_accum._imag);
			  }
		    }
		}
	}
}
