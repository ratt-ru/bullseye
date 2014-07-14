#pragma once

#include <cstdio>
#include <cstdint>
#include <complex>
#include <string>
#include <casa/Quanta/Quantum.h>
#include <cmath>

#include "uvw_coord.h"
#include "baseline_transform_policies.h"
#include "phase_transform_policies.h"
#include "polarization_gridding_policies.h"
#include "convolution_policies.h"


//Enable this if you want some progress printed out:
#define GRIDDER_PRINT_PROGRESS

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
		reference_wavelengths: associated wavelength of each channel used to sample visibilities (wavelength = speed_of_light / channel_frequency)
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
		  typename convolution_policy_type = convolution_policy<convolution_base_type, uvw_base_type, baseline_transformation_policy_type, convolution_precomputed_fir> >
	void grid(polarization_gridding_policy_type & __restrict__ active_polarization_gridding_policy,
		  const baseline_transformation_policy_type & __restrict__ active_baseline_transform_policy,
		  const convolution_policy_type & __restrict__ active_convolution_policy,
		  const uvw_coord<uvw_base_type> * __restrict__ uvw_coords,
		  const bool * __restrict__ flagged_rows,
		  std::size_t nx, std::size_t ny, casa::Quantity cellx, casa::Quantity celly,
		  std::size_t timestamp_count, std::size_t baseline_count, std::size_t channel_count,
		  std::size_t row_count,
		  const reference_wavelengths_base_type *__restrict__ reference_wavelengths){
		/*
		Pg. 138, 145-147, Synthesis Imaging II (Briggs, Schwab & Sramek)
		Scale the UVW coords so that we can image only the primary field of view:
		-(Nx * cellx)/2 < l < (Nx * cellx)/2
		-(Ny * celly)/2 < m < (Ny * celly)/2
		Scaling the uv coordinates will translate to scaling the FFT
		*/
                uvw_base_type u_scale=nx*cellx.getValue("rad");
                uvw_base_type v_scale=ny*celly.getValue("rad");
		
		#ifdef GRIDDER_PRINT_PROGRESS
		//give some indication of progress:
		float progress_step_size = 10.0;
		float next_progress_step = progress_step_size;
		#endif
                for (std::size_t bt = 0; bt < row_count; ++bt){ //this corresponds to the rows of the MS 2.0 MAIN table definition
			#ifdef GRIDDER_PRINT_PROGRESS
			float progress = bt/float(row_count)*100.0f;
			if (progress > next_progress_step){
				printf("%f%%... ",next_progress_step);
				fflush(stdout);
				next_progress_step += progress_step_size;
			}
			#endif
			if (flagged_rows[bt]) continue; //if the entire row is flagged don't continue
                        for (std::size_t c = 0; c < channel_count; ++c){
				/*
				 Get uvw coords and measure the uvw coordinates in terms of wavelength
				*/
                                uvw_coord<uvw_base_type> uvw = uvw_coords[bt];
				uvw._u *= (1/reference_wavelengths[c]);
				uvw._v *= (1/reference_wavelengths[c]);
				
				/*	
				 By default this uvw transform does nothing, but it can be set to do rotate a lw facet to be tangent to a new phase centre		
				 The default phase transform does nothing, but it can be set to rotate the visibilities to a new phase centre in lw / uv faceting.
				 
				 The phase transformation policy is applied to all the polarization terms by the polarization gridding policy. This provides a 
				 neat way of handling multiple polarization options when it comes to gridding. The phase transformation term should be seen 
				 as a scalar Jones term when dealing with 2x2 polarized visibility terms for instance.
				 
				 Refer to Cornwell & Perley (1992), Synthesis Imaging II (1999) and Smirnov I (2011)
				*/
				active_baseline_transform_policy.transform(uvw);
				active_polarization_gridding_policy.transform(bt,c,baseline_count,timestamp_count,channel_count,uvw); //reads and transforms the current visibility
				
				/*
				 Now that all the transformations are done, convert and scale the uv coords down to grid space:
				*/
				uvw._u = uvw._u*u_scale;
				uvw._v = -uvw._v*v_scale;
				
				/*				
				On page 25-26 of Synthesis Imaging II Thompson notes that the correlator output is a measure of the visibility at two points on the
				uv grid. This corresponds to the baseline and the reversed baseline direction: b and -b. The latter represents the complex conjugate of the visibility
				on b. Rather than running though all the visibilities twice we compute uvw, -uvw, V and V*. We should be able to save ourselves some compuation on phase
				shifts, etc by so doing. All gridder policies must reflect this and support gridding both the complex visibility and its conjugate. 
				*/
				uvw_coord<uvw_base_type> uvw_neg = uvw;
				uvw_neg._u *= -1;
				uvw_neg._v *= -1;
				
				active_convolution_policy.convolve(uvw, &polarization_gridding_policy_type::grid_polarization_terms);
 				active_convolution_policy.convolve(uvw_neg, &polarization_gridding_policy_type::grid_polarization_conjugate_terms);
                        }
                }
                
        }
}


