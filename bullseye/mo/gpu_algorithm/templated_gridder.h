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

#include "cu_common.h"
#include "gridding_parameters.h"
#include "baseline_transform_policies.h"
#include "phase_transform_policies.h"
#include "baseline_transform_traits.h"

namespace imaging {
	/*
		This is a gridding kernel following Romeins distribution stategy.
		This should be launched with 
			block dimensions: {THREADS_PER_BLOCK,1,1}
			blocks per grid: {minimum number of blocks required to run baselines*(conv_support_size^2)*num_facets threads,
					  1,1}
	 */
	template <typename active_correlation_gridding_policy,
		  typename active_baseline_transformation_policy,
		  typename active_phase_transformation_policy,
		  typename active_convolution_policy>
	__global__ void templated_gridder(gridding_parameters params){
		size_t tid = cu_indexing_schemes::getGlobalIdx_1D_1D(gridDim,blockIdx,blockDim,threadIdx);
		size_t conv_full_support = (params.conv_support << 1) + 1;
		size_t conv_full_support_sq = conv_full_support * conv_full_support;
		if (tid >= params.num_facet_centres * params.baseline_count * conv_full_support_sq) return;
		size_t conv_theadid_flat_index = tid % conv_full_support_sq;
		size_t my_conv_v = (conv_theadid_flat_index / conv_full_support);
		size_t my_conv_u = (conv_theadid_flat_index % conv_full_support);
		size_t facet_mul_baseline_index = tid / conv_full_support_sq; //this is baseline_id * facet_id
		size_t my_facet_id = facet_mul_baseline_index / params.baseline_count;
		size_t my_baseline = facet_mul_baseline_index % params.baseline_count;
		size_t starting_row_index = params.baseline_starting_indexes[my_baseline];
		//the starting index prescan must be n(n-1)/2 + n + 1 elements long since we need the length of the last baseline
		size_t baseline_num_timestamps = params.baseline_starting_indexes[my_baseline+1] - starting_row_index;
		
		//Scale the IFFT by the simularity theorem to the correct FOV
		uvw_base_type u_scale=params.nx*params.cell_size_x * ARCSEC_TO_RAD;
		uvw_base_type v_scale=-(params.ny*params.cell_size_y * ARCSEC_TO_RAD);
		
		uvw_base_type grid_centre_offset_x = params.nx/2 - params.conv_support + my_conv_u;
		uvw_base_type grid_centre_offset_y = params.ny/2 - params.conv_support + my_conv_v;
		size_t grid_size_in_floats = params.nx * params.ny << 1;
		grid_base_type* facet_output_buffer;
		active_correlation_gridding_policy::compute_facet_grid_ptr(params,my_facet_id,grid_size_in_floats,&facet_output_buffer);
		//Compute the transformation necessary to distort the baseline and phase according to the new facet delay centre (Cornwell & Perley, 1991)
		typename active_baseline_transformation_policy::baseline_transform_type baseline_transformation;
		lmn_coord phase_offset;
		uvw_base_type new_delay_ra;
		uvw_base_type new_delay_dec;
		active_phase_transformation_policy::read_facet_ra_dec(params,my_facet_id,new_delay_ra,new_delay_dec);
		active_baseline_transformation_policy::compute_transformation_matrix(params.phase_centre_ra,params.phase_centre_dec,
										     new_delay_ra,new_delay_dec,baseline_transformation);
		active_phase_transformation_policy::compute_delta_lmn(params.phase_centre_ra,params.phase_centre_dec,
								      new_delay_ra,new_delay_dec,phase_offset);
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
				//either all threads in the filter take this branch or not, its better than doing uneccesary accesses to global memory
				if (!(channel_enabled && row_is_in_field_being_imaged && row_is_in_current_spw) || row_flagged) continue;
				typename active_correlation_gridding_policy::active_trait::vis_type vis;
				typename active_correlation_gridding_policy::active_trait::vis_weight_type vis_weight;
				typename active_correlation_gridding_policy::active_trait::vis_flag_type visibility_flagged;
				active_correlation_gridding_policy::read_corralation_data(params,row,spw,c,vis,visibility_flagged,vis_weight);
				/*read and apply the two corrected jones terms if in faceting mode ( Jp^-1 . X . Jq^H^-1 ) --- either DIE or DDE 
				  assuming small fields of view. Weighting is a scalar and can be apply in any order, so lets just first 
				  apply the corrections*/
				active_correlation_gridding_policy::read_and_apply_antenna_jones_terms(params,row,vis);
				//compute the weighted visibility and promote the flags to integers so that we don't have unnecessary complex logic to deal with gridding
				//all or some of the correlations here
				typename active_correlation_gridding_policy::active_trait::vis_flag_type vis_flagged = !(visibility_flagged);
 				typename active_correlation_gridding_policy::active_trait::vis_weight_type combined_vis_weight = 
					 vis_weight * vector_promotion<int,visibility_base_type>(vector_promotion<bool,int>(vis_flagged));
				uvw._u *= ref_wavelength;
				uvw._v *= ref_wavelength;
				uvw._w *= ref_wavelength;
				//Do phase rotation in accordance with Cornwell & Perley (1992)
				active_phase_transformation_policy::apply_phase_transform(phase_offset,uvw,vis);
				//DO baseline rotation in accordance with Cornwell & Perley (1992)
				active_baseline_transformation_policy::apply_transformation(uvw,baseline_transformation);
				//scale the uv coordinates (measured in wavelengths) to the correct FOV by the fourier simularity theorem (pg 146-148 Synthesis Imaging in Radio Astronomy II)
				uvw._u *= u_scale; 
				uvw._v *= v_scale;
				//account for interpolation error (we select the closest sample from the oversampled convolution filter)
				size_t my_current_u = 0,my_current_v = 0,closest_conv_u = 0,closest_conv_v = 0;
				active_convolution_policy::compute_closest_uv_in_conv_kernel(params,vis,uvw,
											     grid_centre_offset_x,
											     grid_centre_offset_y,
											     my_conv_u,my_conv_v,
											     my_current_u,my_current_v,
											     closest_conv_u,closest_conv_v);
				//if this is the first timestamp for this baseline initialize previous_u and previous_v
				if (t == 0) {
					my_previous_u = my_current_u;
					my_previous_v = my_current_v;
					//already set: normalization_term = 0;
				}
				//if u and v have changed we must dump everything to memory at previous_u and previous_v and reset
				if ((my_current_u != my_previous_u || my_current_v != my_previous_v) && channel_enabled){
					//don't you dare go off the grid:
					if (my_previous_v + conv_full_support < params.ny && my_previous_u + conv_full_support < params.nx &&
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
				//read convolution weight and then multiply-add into the accumulator
				active_convolution_policy::convolve(params,
								    uvw,vis,
								    my_grid_accum,
								    normalization_term,
								    combined_vis_weight,
								    closest_conv_u,closest_conv_v);
			    } //time
			    //Okay time to dump everything since the last uv shift
			    if (my_previous_u + conv_full_support < params.ny && my_previous_u + conv_full_support  < params.nx &&
				my_previous_v < params.ny && my_previous_u < params.nx && channel_enabled){
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
												     my_conv_u,
												     my_conv_v);
			    }//dumping last accumulation
			}//channel
		    }//spw
		}
	}
}
