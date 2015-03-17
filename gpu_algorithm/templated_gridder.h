#pragma once

#include "cu_common.h"
#include "gridding_parameters.h"
#include "baseline_transform_policies.h"
#include "phase_transform_policies.h"

namespace imaging {
	/*
		This is a gridding kernel following Romeins distribution stategy.
		This should be launched with 
			block dimensions: {THREADS_PER_BLOCK,1,1}
			blocks per grid: {minimum number of blocks required to run baselines*conv_support_size*num_facets^^2 threads,
					  1,1}
	 */
	template <typename active_correlation_gridding_policy,
		  typename active_baseline_transformation_policy,
		  typename active_phase_transformation>
	__global__ void templated_gridder(gridding_parameters params){
		size_t tid = cu_indexing_schemes::getGlobalIdx_1D_1D(gridDim,blockIdx,blockDim,threadIdx);
		size_t conv_full_support = (params.conv_support << 1) + 1;
		size_t conv_full_support_sq = conv_full_support * conv_full_support;
		size_t padded_conv_full_support = conv_full_support + 2; //remember we need to reserve some of the support for +/- frac on both sides
		if (tid >= params.num_facet_centres * params.baseline_count * conv_full_support_sq) return;
		size_t conv_theadid_flat_index = tid % conv_full_support_sq;
		size_t my_conv_v = (conv_theadid_flat_index / conv_full_support) + 1;
		size_t my_conv_u = (conv_theadid_flat_index % conv_full_support) + 1;
		size_t facet_mul_baseline_index = tid / conv_full_support_sq; //this is baseline_id * facet_id
		size_t my_facet_id = facet_mul_baseline_index / params.baseline_count;
		size_t my_baseline = facet_mul_baseline_index % params.baseline_count;
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
		grid_base_type* facet_output_buffer;
		active_correlation_gridding_policy::compute_facet_grid_ptr(params,my_facet_id,grid_size_in_floats,&facet_output_buffer);
		//Compute the transformation necessary to distort the baseline and phase according to the new facet delay centre (Cornwell & Perley, 1991)
		baseline_rotation_mat baseline_transformation;
		lmn_coord phase_offset;
		uvw_base_type new_delay_ra;
		uvw_base_type new_delay_dec;
		active_phase_transformation::read_facet_ra_dec(params,my_facet_id,new_delay_ra,new_delay_dec);
		active_baseline_transformation_policy::compute_transformation_matrix(params.phase_centre_ra,params.phase_centre_dec,
										     new_delay_ra,new_delay_dec,baseline_transformation);
		active_phase_transformation::compute_delta_lmn(params.phase_centre_ra,params.phase_centre_dec,
							       new_delay_ra,new_delay_dec,phase_offset);
		//load the convolution filter into shared memory
		extern __shared__ convolution_base_type shared_conv[];
		if (threadIdx.x == 0){
		  size_t fir_ubound = ((params.conv_oversample * padded_conv_full_support));
		  
		  for (size_t x = 0; x < fir_ubound; ++x){
		    shared_conv[x] = params.conv[x];
		  }
		}
		__syncthreads(); //wait for the first thread to put the entire filter into shared memory
		
// 		size_t channel_loop_ubound = params.channel_count >> 1;
// 		size_t channel_loop_rem_lbound = channel_loop_ubound << 1;
		//we must keep seperate accumulators per channel, so we need to bring these loops outward (contrary to Romein's paper)
		{
		    typename active_correlation_gridding_policy::active_trait::accumulator_type my_grid_accum;
		    for (size_t spw = 0; spw < params.spw_count; ++spw){
			for (size_t c = 0; c < params.channel_count; ++c){	
			    my_grid_accum = active_correlation_gridding_policy::active_trait::vis_type::zero();
			    int my_previous_u = 0;
			    int my_previous_v = 0;
			    typename active_correlation_gridding_policy::active_trait::normalization_accumulator_type normalization_term = 0;
			    //read all the stuff that is only dependent on the current spw and channel
			    size_t flat_indexed_spw_channel = spw * params.channel_count + c;
			    bool channel_enabled = params.enabled_channels[flat_indexed_spw_channel];
			    size_t channel_grid_index;
			    active_correlation_gridding_policy::read_channel_grid_index(params,flat_indexed_spw_channel,channel_grid_index);
			    reference_wavelengths_base_type ref_wavelength = 1 / params.reference_wavelengths[flat_indexed_spw_channel];
			    for (size_t t = 0; t < baseline_num_timestamps; ++t){
				size_t row = starting_row_index + t;
				size_t row_spw_id = params.spw_index_array[row];
				bool row_is_in_current_spw = row_spw_id == spw;
				//read all the data we need for gridding
				imaging::uvw_coord<uvw_base_type> uvw = params.uvw_coords[row];
				bool row_flagged = params.flagged_rows[row];
				bool row_is_in_field_being_imaged = (params.field_array[row] == params.imaging_field);
				typename active_correlation_gridding_policy::active_trait::vis_type vis;
				typename active_correlation_gridding_policy::active_trait::vis_weight_type vis_weight;
				typename active_correlation_gridding_policy::active_trait::vis_flag_type visibility_flagged;
				active_correlation_gridding_policy::read_corralation_data(params,row,spw,c,vis,visibility_flagged,vis_weight);
				/*read and apply the two corrected jones terms if in faceting mode ( Jp^-1 . X . Jq^H^-1 ) --- either DIE or DDE 
				  assuming small fields of view. Weighting is a scalar and can be apply in any order, so lets just first 
				  apply the corrections*/
				active_correlation_gridding_policy::read_and_apply_antenna_jones_terms(params,row,vis);
				//compute the weighted visibility and promote the flags to integers so that we don't have unnecessary branch diversion here
				typename active_correlation_gridding_policy::active_trait::vis_flag_type vis_flagged = !(visibility_flagged || row_flagged) && 
														       channel_enabled && row_is_in_field_being_imaged &&
														       row_is_in_current_spw;
 				typename active_correlation_gridding_policy::active_trait::vis_weight_type combined_vis_weight = 
					 vis_weight * vector_promotion<int,visibility_base_type>(vector_promotion<bool,int>(vis_flagged));
				uvw._u *= ref_wavelength;
				uvw._v *= ref_wavelength;
				uvw._w *= ref_wavelength;
				//Do phase rotation in accordance with Cornwell & Perley (1992)
				active_phase_transformation::apply_phase_transform(phase_offset,uvw,vis);
				//DO baseline rotation in accordance with Cornwell & Perley (1992)
				active_baseline_transformation_policy::apply_transformation(uvw,baseline_transformation);
				//scale the uv coordinates (measured in wavelengths) to the correct FOV by the fourier simularity theorem (pg 146-148 Synthesis Imaging in Radio Astronomy II)
				uvw._u *= u_scale; 
				uvw._v *= v_scale;
				//account for interpolation error (we select the closest sample from the oversampled convolution filter)
				uvw_base_type cont_current_u = uvw._u + grid_centre_offset_x;
				uvw_base_type cont_current_v = uvw._v + grid_centre_offset_y;
				int my_current_u = cont_current_u;
				int my_current_v = cont_current_v;
				uvw_base_type frac_u = -cont_current_u + (uvw_base_type)my_current_u;
				uvw_base_type frac_v = -cont_current_v + (uvw_base_type)my_current_v;
				size_t closest_conv_u = ((uvw_base_type)my_conv_u + frac_u)*params.conv_oversample;
				size_t closest_conv_v = ((uvw_base_type)my_conv_v + frac_v)*params.conv_oversample;
				//if this is the first timestamp for this baseline initialize previous_u and previous_v
				if (t == 0) {
					my_previous_u = my_current_u;
					my_previous_v = my_current_v;
					//already set: normalization_term = 0;
				}
				//if u and v have changed we must dump everything to memory at previous_u and previous_v and reset
				if ((my_current_u != my_previous_u || my_current_v != my_previous_v) && channel_enabled){
					//don't you dare go off the grid:
					if (my_previous_v + conv_full_support  < params.ny && my_previous_u + conv_full_support  < params.nx &&
					    my_previous_v < params.ny && my_previous_u < params.nx){
						active_correlation_gridding_policy::grid_visibility(facet_output_buffer,
												    grid_size_in_floats,
												    params.nx,
												    channel_grid_index,
												    params.number_of_polarization_terms_being_gridded,
												    my_previous_u,
												    my_previous_v,
												    my_grid_accum
												   );
					}
					my_grid_accum = active_correlation_gridding_policy::active_trait::vis_type::zero();
					my_previous_u = my_current_u;
					my_previous_v = my_current_v;
				}
				//Lets read the convolution weights from the the precomputed filter
				convolution_base_type conv_weight = shared_conv[closest_conv_u] * shared_conv[closest_conv_v];
				//then multiply-add into the accumulator
				typename active_correlation_gridding_policy::active_trait::vis_weight_type conv_weighted_vis_weight = combined_vis_weight * conv_weight;
				my_grid_accum += vis * conv_weighted_vis_weight;
				normalization_term += vector_promotion<visibility_weights_base_type,normalization_base_type>(conv_weighted_vis_weight);
			    } //time
			    //Okay time to dump everything since the last uv shift
			    if (my_previous_u + conv_full_support < params.ny && my_previous_u + conv_full_support  < params.nx &&
				my_previous_v < params.ny && my_previous_u < params.nx){
				active_correlation_gridding_policy::grid_visibility(facet_output_buffer,
										    grid_size_in_floats,
										    params.nx,
										    channel_grid_index,
										    params.number_of_polarization_terms_being_gridded,
										    my_previous_u,
										    my_previous_v,
										    my_grid_accum);
				active_correlation_gridding_policy::update_normalization_accumulator(params,normalization_term,
												     my_facet_id,
												     channel_grid_index,
												     conv_full_support,
												     my_conv_u - 1,
												     my_conv_v - 1);
			    }//dumping last accumulation
			}//channel
		    }//spw
		}
	}
}
