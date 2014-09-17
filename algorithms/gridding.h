#pragma once

#include <cstdio>
#include <cstdint>
#include <complex>
#include <string>
#include <casa/Quanta/Quantum.h>
#include <cmath>
#include <cfenv>

#include "uvw_coord.h"
#include "baseline_transform_policies.h"
#include "phase_transform_policies.h"
#include "polarization_gridding_policies.h"
#include "convolution_policies.h"


//Enable this if you want some progress printed out:
//#define GRIDDER_PRINT_PROGRESS

namespace imaging {
	/**
		Convolutional gridding algorithm
		Arguements:
		active_polarization_gridding_policy: precreated policy describing how to grid polarization
		active_phase_transform_policy: precreated phase transform policy
		active_convolution_policy: precreated policy handling how the convolution itself is performed
		uvw_coord: set of associated uvw coordinates (measured in arcsec, ***NOT*** in wavelength (ie. meters))
		nx,ny: size of the pre-allocated buffer
		cellx,celly: size of the cells (in arcsec)
		timestamp_count, baseline_count, channel_count, polarization_term_count: (integral counts as specified)
		row_count: number of measurement set rows being imaged (this can be equal to timestamp_count*baseline_count if the entire measurement set is read into memory)
		reference_wavelengths: associated wavelength of each channel per spectral window used to sample visibilities (wavelength = speed_of_light / channel_frequency)
		field_array: this corresponds to the FIELD_ID column of an ms, and can be used to identify where each baseline is pointing at a particular time (see also "imaging_field")
		imaging_field: this is the identifier of the field (pointing) currently being imaged. Only contributions from baselines with this field identifier will be gridded.
		spw_index_array: this array specifies which reference frequency to select (reference_wavelengths has dimensions no_spws * no_channels)
			PRECONDITIONS:
			1. timestamp_count x baseline_count x channel_count x polarization_term_count <= ||visibilities||
			2. ||flagged rows + unflagged rows|| == ||visibilities array||
	*/
	template  <typename visibility_base_type, typename uvw_base_type, 
		   typename reference_wavelengths_base_type, typename convolution_base_type,
		   typename weights_base_type,
		   typename grid_base_type,
		   typename baseline_transformation_policy_type = baseline_transform_policy<uvw_base_type, transform_disable_facet_rotation>,
		   typename polarization_gridding_policy_type = polarization_gridding_policy<visibility_base_type,uvw_base_type,
											     weights_base_type,
											     convolution_base_type,
											     grid_base_type,
											     phase_transform_policy<visibility_base_type, 
														    uvw_base_type, 
														    transform_disable_phase_rotation>,
											     gridding_4_pol>,
		  typename convolution_policy_type = convolution_policy<convolution_base_type, uvw_base_type, 
									grid_base_type, baseline_transformation_policy_type, 
									convolution_precomputed_fir> >
	void grid(polarization_gridding_policy_type & __restrict__ active_polarization_gridding_policy,
		  const baseline_transformation_policy_type & __restrict__ active_baseline_transform_policy,
		  const convolution_policy_type & __restrict__ active_convolution_policy,
		  const uvw_coord<uvw_base_type> * __restrict__ uvw_coords,
		  const bool * __restrict__ flagged_rows,
		  std::size_t nx, std::size_t ny, casa::Quantity cellx, casa::Quantity celly,
		  std::size_t channel_count,
		  std::size_t row_count,
		  const reference_wavelengths_base_type *__restrict__ reference_wavelengths,
		  const unsigned int * __restrict__ field_array,
		  unsigned int imaging_field,
		  const unsigned int * __restrict__ spw_index_array,
		  const std::size_t * __restrict__ channel_grid_indicies,
		  const bool * __restrict__ enabled_channels
 		){
		/*
		Pg. 138, 145-147, Synthesis Imaging II (Briggs, Schwab & Sramek)
		Scale the UVW coords so that we can image only the primary field of view:
		-(Nx * cellx)/2 < l < (Nx * cellx)/2
		-(Ny * celly)/2 < m < (Ny * celly)/2
		Scaling the uv coordinates will translate to scaling the FFT
		*/
		std::fesetround(FE_TONEAREST);
                uvw_base_type u_scale=nx*cellx.getValue("rad");
                uvw_base_type v_scale=ny*celly.getValue("rad");
		auto uv_scale = uvw_coord<uvw_base_type>(u_scale,-v_scale);
		#ifdef GRIDDER_PRINT_PROGRESS
		//give some indication of progress:
		float progress_step_size = 10.0;
		float next_progress_step = progress_step_size;
		#endif
		#pragma omp parallel for
                for (std::size_t bt = 0; bt < row_count; ++bt){ //this corresponds to the rows of the MS 2.0 MAIN table definition
			#ifdef GRIDDER_PRINT_PROGRESS
			float progress = bt/float(row_count)*100.0f;
			if (progress > next_progress_step){
				printf("%f%%... ",next_progress_step);
				fflush(stdout);
				next_progress_step += progress_step_size;
			}
			#endif
			if (field_array[bt] != imaging_field) continue; //We only image contributions from those antennae actually pointing to the field in question
			if (flagged_rows[bt]) continue; //if the entire row is flagged don't continue
			std::size_t spw_index = spw_index_array[bt];
			unsigned int current_spw_offset = spw_index*channel_count;
                        for (std::size_t c = 0; c < channel_count; ++c){
				if (!enabled_channels[current_spw_offset + c]) continue;
				/*
				 Get uvw coords
				*/
				uvw_coord<uvw_base_type> uvw = uvw_coords[bt];
				
				/*
				 * Now measure the uvw coordinates in terms of wavelength
				 */
				reference_wavelengths_base_type wavelength = (reference_wavelengths_base_type)(1.0/reference_wavelengths[current_spw_offset + c]);
				uvw *= wavelength;
				/*	
				 By default this uvw transform does nothing, but it can be set to do rotate a lw facet to be tangent to a new phase centre		
				 The default phase transform does nothing, but it can be set to rotate the visibilities to a new phase centre in lw / uv faceting.
				 
				 The phase transformation policy is applied to all the polarization terms by the polarization gridding policy. This provides a 
				 neat way of handling multiple polarization options when it comes to gridding. The phase transformation term should be seen 
				 as a scalar Jones term when dealing with 2x2 polarized visibility terms for instance.
				 
				 Refer to Cornwell & Perley (1992), Synthesis Imaging II (1999) and Smirnov I (2011)
				*/
				typename polarization_gridding_policy_type::trait_type::pol_vis_type vis;
				active_polarization_gridding_policy.transform(bt,spw_index,c,uvw,vis); //reads and transforms the current visibility
				//as per Cornwell and Perley... then we rotate the uvw coordinate...
				active_baseline_transform_policy.transform(uvw);
				/*
				 Now that all the transformations are done, convert and scale the uv coords down to the desired FOV (this scales the IFFT by the simularity theorem):
				*/
				uvw *= uv_scale;
				
				/*				
				On page 25-26 of Synthesis Imaging II Thompson notes that the correlator output is a measure of the visibility at two points on the
				uv grid. This corresponds to the baseline and the reversed baseline direction: b and -b. The latter represents the complex conjugate of the visibility
				on b. This hermitian symetric component need not be gridded and relates to a 2x improvement in runtime
				*/
				active_convolution_policy.convolve(uvw, vis, channel_grid_indicies[current_spw_offset + c]);
                        }
                }
                
        }
}


