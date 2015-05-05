#pragma once

#include <cfenv>
#include "gridding_parameters.h"
#include "baseline_transform_policies.h"
#include "phase_transform_policies.h"
#include "baseline_transform_traits.h"
#include "correlation_gridding_policies.h"
#include "convolution_policies.h"
#include "correlation_gridding_traits.h"

namespace imaging {
	template <typename active_correlation_gridding_policy,
		  typename active_baseline_transformation_policy,
		  typename active_phase_transformation,
		  typename active_convolution_policy>
	__global__ void templated_gridder(gridding_parameters & params){
		active_convolution_policy::set_required_rounding_operation();
		size_t conv_full_support = (params.conv_support << 1) + 1;
		size_t conv_full_support_sq = conv_full_support * conv_full_support;
		size_t padded_conv_full_support = conv_full_support + 2; //remember we need to reserve some of the support for +/- frac on both sides
		
		//Scale the IFFT by the simularity theorem to the correct FOV
		uvw_base_type u_scale=params.nx*params.cell_size_x * ARCSEC_TO_RAD;
		uvw_base_type v_scale=-(params.ny*params.cell_size_y * ARCSEC_TO_RAD);
		
		uvw_base_type conv_offset = (padded_conv_full_support) / 2.0; 
		uvw_base_type grid_centre_offset_x = params.nx/2 - conv_offset;
		uvw_base_type grid_centre_offset_y = params.ny/2 - conv_offset;
		size_t grid_size_in_floats = params.nx * params.ny << 1;
		
		#pragma omp parallel for
		for (size_t my_facet_id = 0; my_facet_id < params.num_facet_centres; ++my_facet_id){
		  grid_base_type* facet_output_buffer;
		  active_correlation_gridding_policy::compute_facet_grid_ptr(params,my_facet_id,grid_size_in_floats,&facet_output_buffer);
		  //Compute the transformation necessary to distort the baseline and phase according to the new facet delay centre (Cornwell & Perley, 1991)
		  typename active_baseline_transformation_policy::baseline_transform_type baseline_transformation;
		  lmn_coord phase_offset;
		  uvw_base_type new_delay_ra;
		  uvw_base_type new_delay_dec;
		  active_phase_transformation::read_facet_ra_dec(params,my_facet_id,new_delay_ra,new_delay_dec);
		  active_baseline_transformation_policy::compute_transformation_matrix(params.phase_centre_ra,params.phase_centre_dec,
										      new_delay_ra,new_delay_dec,baseline_transformation);
		  active_phase_transformation::compute_delta_lmn(params.phase_centre_ra,params.phase_centre_dec,
								 new_delay_ra,new_delay_dec,phase_offset);
		  
		  for (size_t row = 0; row < params.row_count; ++row){
			size_t spw = params.spw_index_array[row];
			for (size_t c = 0; c < params.channel_count; ++c){	
			    //read all the stuff that is only dependent on the current spw and channel    
			    size_t flat_indexed_spw_channel = spw * params.channel_count + c;
			    bool channel_enabled = params.enabled_channels[flat_indexed_spw_channel];
			    size_t channel_grid_index;
			    active_correlation_gridding_policy::read_channel_grid_index(params,flat_indexed_spw_channel,channel_grid_index);
			    reference_wavelengths_base_type ref_wavelength = 1 / params.reference_wavelengths[flat_indexed_spw_channel];
			    
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
			    active_correlation_gridding_policy::read_and_apply_antenna_jones_terms(params,row,my_facet_id,spw,c,vis);
			    //compute the weighted visibility and promote the flags to integers so that we don't have unnecessary branch diversion here
			    typename active_correlation_gridding_policy::active_trait::vis_flag_type vis_flagged = !(visibility_flagged || row_flagged) && 
														     channel_enabled && row_is_in_field_being_imaged;
			    typename active_correlation_gridding_policy::active_trait::vis_weight_type combined_vis_weight = vis_weight * 
												       vector_promotion<int,visibility_base_type>(vector_promotion<bool,int>(vis_flagged));
			    vis = vis * combined_vis_weight;
			    uvw._u *= ref_wavelength;
			    uvw._v *= ref_wavelength;
			    uvw._w *= ref_wavelength;
			    //Do phase rotation in accordance with Cornwell & Perley (1992)
			    active_phase_transformation::apply_phase_transform(phase_offset,uvw,vis);
			    //DO baseline rotation in accordance with Cornwell & Perley (1992) / Greisen 2009 --- latter results in coplanar facets
			    active_baseline_transformation_policy::apply_transformation(uvw,baseline_transformation);
			    //scale the uv coordinates (measured in wavelengths) to the correct FOV by the fourier simularity theorem (pg 146-148 Synthesis Imaging in Radio Astronomy II)
			    uvw._u *= u_scale; 
			    uvw._v *= v_scale;
			    typename active_correlation_gridding_policy::active_trait::normalization_accumulator_type normalization_term = 0;
			    active_convolution_policy::convolve(params,grid_centre_offset_x,grid_centre_offset_y,
								facet_output_buffer,channel_grid_index,grid_size_in_floats,
								conv_full_support,padded_conv_full_support,uvw,vis,normalization_term);
			    normalization_term = vector_promotion<visibility_weights_base_type,normalization_base_type>(combined_vis_weight * normalization_term._x);
			    active_correlation_gridding_policy::store_normalization_term(params,channel_grid_index,my_facet_id,
											 normalization_term);
			}//channel
		  }//row
		}//facet
	}
}