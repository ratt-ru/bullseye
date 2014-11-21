#include <string>
#include <cstdio>
#include <casa/Quanta/Quantum.h>
#include <numeric>
#include <fftw3.h>
#include <typeinfo>
#include <thread>
#include <future>

#include "omp.h"
#include "wrapper.h"
#include "timer.h"
#include "uvw_coord.h"
#include "baseline_transform_policies.h"
#include "phase_transform_policies.h"
#include "polarization_gridding_policies.h"
#include "convolution_policies.h"
#include "gridding.h"
#include "fft_shift_utils.h"

#ifdef SHOULD_DO_32_BIT_FFT
  typedef fftwf_plan fftw_plan_type;
#else
  typedef fftw_plan fftw_plan_type;
#endif
extern "C" {
    fftw_plan_type fft_plan;
    fftw_plan_type fft_psf_plan;
    utils::timer gridding_timer;
    utils::timer inversion_timer;
    std::future<void> gridding_future;
    double get_gridding_walltime() {
      return gridding_timer.duration();
    }
    double get_inversion_walltime() {
      return inversion_timer.duration();
    }
    void gridding_barrier() {
        if (gridding_future.valid())
            gridding_future.get(); //Block until result becomes available
    }
    void initLibrary(gridding_parameters & params){
      printf("-----------------------------------------------\n"
	     "      Backend: Multithreaded CPU Library       \n\n");	     
      printf(" >Number of cores available: %d\n",omp_get_num_procs());
      printf(" >Number of threads being used: %d\n",omp_get_max_threads());
      printf("-----------------------------------------------\n");
      int dims[] = {(int)params.nx,(int)params.ny};
      #ifdef SHOULD_DO_32_BIT_FFT
	fft_plan = fftwf_plan_many_dft(2,(int*)&dims,
				       params.cube_channel_dim_size,
				       (fftwf_complex *)params.output_buffer,(int*)&dims,
				       1,(int)(params.nx*params.ny),
				       (fftwf_complex *)params.output_buffer,(int*)&dims,
				       1,(int)(params.nx*params.ny),
				       FFTW_BACKWARD,FFTW_ESTIMATE | FFTW_UNALIGNED);
	fft_psf_plan = fftwf_plan_many_dft(2,(int*)&dims,
					   params.sampling_function_channel_count * params.num_facet_centres,
					   (fftwf_complex *)params.sampling_function_buffer,(int*)&dims,
					   1,(int)(params.nx*params.ny),
					   (fftwf_complex *)params.sampling_function_buffer,(int*)&dims,
					   1,(int)(params.nx*params.ny),
					   FFTW_BACKWARD,FFTW_ESTIMATE | FFTW_UNALIGNED);
      #else
	fft_plan = fftw_plan_many_dft(2,(int*)&dims,
				      params.cube_channel_dim_size,
				      (fftw_complex *)params.output_buffer,(int*)&dims,
				      1,(int)(params.nx*params.ny),
				      (fftw_complex *)params.output_buffer,(int*)&dims,
				      1,(int)(params.nx*params.ny),
				      FFTW_BACKWARD,FFTW_ESTIMATE | FFTW_UNALIGNED);
	fft_psf_plan = fftw_plan_many_dft(2,(int*)&dims,
					  params.sampling_function_channel_count * params.num_facet_centres,
					  (fftw_complex *)params.sampling_function_buffer,(int*)&dims,
					  1,(int)(params.nx*params.ny),
					  (fftw_complex *)params.sampling_function_buffer,(int*)&dims,
					  1,(int)(params.nx*params.ny),
					  FFTW_BACKWARD,FFTW_ESTIMATE | FFTW_UNALIGNED);
      #endif
    }
    void releaseLibrary(){
      gridding_barrier();
      #ifdef SHOULD_DO_32_BIT_FFT
	fftwf_destroy_plan(fft_plan);
	fftwf_destroy_plan(fft_psf_plan);
      #else
	fftw_destroy_plan(fft_plan);
	fftw_destroy_plan(fft_psf_plan);
      #endif
    }
    void weight_uniformly(gridding_parameters & params){
      #define EPSILON 0.0000001f
      gridding_barrier();
      for (std::size_t f = 0; f < params.num_facet_centres; ++f)
	for (std::size_t g = 0; g < params.cube_channel_dim_size; ++g)
	  for (std::size_t y = 0; y < params.ny; ++y)
	    for (std::size_t x = 0; x < params.nx; ++x){
		grid_base_type count = EPSILON;
		//accumulate all the sampling functions that contribute to the current grid
		for (std::size_t c = 0; c < params.sampling_function_channel_count; ++c)
		    count += (int)(params.channel_grid_indicies[c] == g) *
			     real(params.sampling_function_buffer[((f*params.sampling_function_channel_count + c)*params.ny+y)*params.nx + x]);
		count = 1/count;
		//and apply to the continuous block of nx*ny*cube_channel grids (any temporary correlation term buffers should have been collapsed by this point)
 		params.output_buffer[((f*params.cube_channel_dim_size*params.number_of_polarization_terms_being_gridded + g)*params.ny+y)*params.nx+x] *= count;
	    }
    }
    void finalize(gridding_parameters & params){
	gridding_barrier();
	inversion_timer.start();
	std::size_t offset = params.nx*params.ny*params.cube_channel_dim_size*params.number_of_polarization_terms_being_gridded;
	#ifdef SHOULD_DO_32_BIT_FFT
	  for (std::size_t f = 0; f < params.num_facet_centres; ++f) {
	    utils::ifftshift(params.output_buffer + f*offset,params.nx,params.ny,params.cube_channel_dim_size);
	    fftwf_execute_dft(fft_plan,
			      (fftwf_complex *)(params.output_buffer + f*offset),
			      (fftwf_complex *)(params.output_buffer + f*offset));
	    utils::fftshift(params.output_buffer + f*offset,params.nx,params.ny,params.cube_channel_dim_size);
	  }
	#else
	  for (std::size_t f = 0; f < params.num_facet_centres; ++f) {
	    utils::ifftshift(params.output_buffer + f*offset,params.nx,params.ny,params.cube_channel_dim_size);
	    fftw_execute_dft(fft_plan,
			   (fftw_complex *)(params.output_buffer + f*offset),
			   (fftw_complex *)(params.output_buffer + f*offset));
	    utils::fftshift(params.output_buffer + f*offset,params.nx,params.ny,params.cube_channel_dim_size);
	  }
	#endif
	/*
	 * We'll be storing 32 bit real fits files so ignore all the imaginary components and cast whatever the grid was to float32
	 */
	{
	  grid_base_type * __restrict__ grid_ptr_gridtype = (grid_base_type *)params.output_buffer;
	  float * __restrict__ grid_ptr_single = (float *)params.output_buffer;
	  std::size_t image_size = (params.nx*params.ny);
	  for (std::size_t f = 0; f < params.num_facet_centres; ++f) {
	      std::size_t casting_lbound = offset*f;
	      std::size_t casting_ubound = casting_lbound + params.nx*params.ny*params.cube_channel_dim_size;
	      for (std::size_t i = casting_lbound; i < casting_ubound; ++i){
		  std::size_t detapering_flat_index = i % image_size;
		  grid_ptr_single[i] = (float)(grid_ptr_gridtype[i*2] / params.detapering_buffer[detapering_flat_index]); //extract all the reals
	      }
	  }
	}	
	inversion_timer.stop();
    }
    void finalize_psf(gridding_parameters & params){
	gridding_barrier();
	inversion_timer.start();
	std::size_t offset = params.nx*params.ny*params.num_facet_centres*params.sampling_function_channel_count;
	#ifdef SHOULD_DO_32_BIT_FFT
	  for (std::size_t f = 0; f < params.num_facet_centres; ++f) 
	    utils::ifftshift(params.sampling_function_buffer + f*offset,params.nx,params.ny,params.sampling_function_channel_count);
	  fftwf_execute(fft_psf_plan);
 	  for (std::size_t f = 0; f < params.num_facet_centres; ++f) 
	    utils::fftshift(params.sampling_function_buffer + f*offset,params.nx,params.ny,params.sampling_function_channel_count);
	#else
	  for (std::size_t f = 0; f < params.num_facet_centres; ++f) 
	    utils::ifftshift(params.sampling_function_buffer + f*offset,params.nx,params.ny,params.sampling_function_channel_count);
	  fftw_execute(fft_psf_plan);
 	  for (std::size_t f = 0; f < params.num_facet_centres; ++f) 
	    utils::fftshift(params.sampling_function_buffer + f*offset,params.nx,params.ny,params.sampling_function_channel_count);
	#endif
	/*
	 * We'll be storing 32 bit real fits files so ignore all the imaginary components and cast whatever the grid was to float32
	 */
	{
	  grid_base_type * __restrict__ grid_ptr_gridtype = (grid_base_type *)params.sampling_function_buffer;
	  float * __restrict__ grid_ptr_single = (float *)params.sampling_function_buffer;
	  std::size_t image_size = (params.nx*params.ny);
	  for (std::size_t f = 0; f < params.num_facet_centres; ++f) {
	      std::size_t casting_lbound = offset*f;
	      std::size_t casting_ubound = casting_lbound + params.nx*params.ny*params.sampling_function_channel_count;
	      for (std::size_t i = casting_lbound; i < casting_ubound; ++i){
		  std::size_t detapering_flat_index = i % image_size;
		  grid_ptr_single[i] = (float)(grid_ptr_gridtype[i*2] / params.detapering_buffer[detapering_flat_index]); //extract all the reals
	      }
	  }
	}
	inversion_timer.stop();
    }
    void grid_single_pol(gridding_parameters & params) {
        gridding_barrier();
        gridding_future = std::async(std::launch::async, [&params] () {
	    gridding_timer.start();
            using namespace imaging;
            printf("GRIDDING...\n");
            typedef baseline_transform_policy<uvw_base_type, transform_disable_facet_rotation> baseline_transform_policy_type;
            typedef phase_transform_policy<visibility_base_type,
                    uvw_base_type,
                    transform_disable_phase_rotation> phase_transform_policy_type;
            typedef polarization_gridding_policy<visibility_base_type, uvw_base_type,
                    visibility_weights_base_type, convolution_base_type, grid_base_type,
                    phase_transform_policy_type, gridding_single_pol> polarization_gridding_policy_type;
            typedef convolution_policy<convolution_base_type,uvw_base_type,grid_base_type,
                    polarization_gridding_policy_type, convolution_precomputed_fir> convolution_policy_type;

            baseline_transform_policy_type uvw_transform; //standard: no uvw rotation
            phase_transform_policy_type phase_transform; //standard: no phase rotation

            polarization_gridding_policy_type polarization_policy(phase_transform,
                    params.output_buffer,
                    params.visibilities,
                    params.visibility_weights,
                    params.flags,
                    params.number_of_polarization_terms,
                    params.polarization_index,
                    params.channel_count);
            convolution_policy_type convolution_policy(params.nx,params.ny,1,params.conv_support,params.conv_oversample,
                    params.conv, polarization_policy);

            imaging::grid<visibility_base_type,uvw_base_type,
                    reference_wavelengths_base_type,convolution_base_type,
                    visibility_weights_base_type,grid_base_type,
                    baseline_transform_policy_type,
                    polarization_gridding_policy_type,
                    convolution_policy_type>
                    (polarization_policy,uvw_transform,convolution_policy,
                     params.uvw_coords,
                     params.flagged_rows,
                     params.nx,params.ny,
                     casa::Quantity(params.cell_size_x,"arcsec"),
                     casa::Quantity(params.cell_size_y,"arcsec"),
                     params.channel_count,
                     params.row_count,params.reference_wavelengths,params.field_array,
                     params.imaging_field,params.spw_index_array,
		     params.channel_grid_indicies,
		     params.enabled_channels);
	    gridding_timer.stop();
        });
    }
    void facet_single_pol(gridding_parameters & params) {
        gridding_barrier();
        gridding_future = std::async(std::launch::async, [&params] () {
	    gridding_timer.start();
            using namespace imaging;
            size_t no_facet_pixels = params.nx*params.ny;
            for (size_t facet_index = 0; facet_index < params.num_facet_centres; ++facet_index) {
                uvw_base_type new_phase_ra = params.facet_centres[2*facet_index];
                uvw_base_type new_phase_dec = params.facet_centres[2*facet_index + 1];
                printf("FACETING SINGLE (%f,%f,%f,%f) %lu / %lu...\n",params.phase_centre_ra,params.phase_centre_dec,new_phase_ra,new_phase_dec,facet_index+1, params.num_facet_centres);

                typedef imaging::baseline_transform_policy<uvw_base_type,
                        transform_facet_lefthanded_ra_dec> baseline_transform_policy_type;
                typedef imaging::phase_transform_policy<visibility_base_type,
                        uvw_base_type,
                        transform_enable_phase_rotation_lefthanded_ra_dec> phase_transform_policy_type;
                typedef imaging::polarization_gridding_policy<visibility_base_type, uvw_base_type,
                        visibility_weights_base_type, convolution_base_type, grid_base_type,
                        phase_transform_policy_type, gridding_single_pol> polarization_gridding_policy_type;
                typedef imaging::convolution_policy<convolution_base_type,uvw_base_type,grid_base_type,
                        polarization_gridding_policy_type,convolution_precomputed_fir> convolution_policy_type;
                baseline_transform_policy_type uvw_transform(0,0,casa::Quantity(params.phase_centre_ra,"arcsec"),casa::Quantity(params.phase_centre_dec,"arcsec"),
                        casa::Quantity(new_phase_ra,"arcsec"),casa::Quantity(new_phase_dec,"arcsec")); //lm faceting
                phase_transform_policy_type phase_transform(casa::Quantity(params.phase_centre_ra,"arcsec"),casa::Quantity(params.phase_centre_dec,"arcsec"),
                        casa::Quantity(new_phase_ra,"arcsec"),casa::Quantity(new_phase_dec,"arcsec")); //lm faceting

                polarization_gridding_policy_type polarization_policy(phase_transform,
                        params.output_buffer + no_facet_pixels*params.cube_channel_dim_size*facet_index,
                        params.visibilities,
                        params.visibility_weights,
                        params.flags,
                        params.number_of_polarization_terms,
                        params.polarization_index,
                        params.channel_count);
                convolution_policy_type convolution_policy(params.nx,params.ny,1,
                        params.conv_support,params.conv_oversample,
                        params.conv, polarization_policy);
                imaging::grid<visibility_base_type,uvw_base_type,
                        reference_wavelengths_base_type,convolution_base_type,
                        visibility_weights_base_type,grid_base_type,
                        baseline_transform_policy_type,
                        polarization_gridding_policy_type,
                        convolution_policy_type>(polarization_policy,uvw_transform,convolution_policy,
                                                 params.uvw_coords,
                                                 params.flagged_rows,
                                                 params.nx,params.ny,
                                                 casa::Quantity(params.cell_size_x,"arcsec"),casa::Quantity(params.cell_size_y,"arcsec"),
                                                 params.channel_count,
                                                 params.row_count,params.reference_wavelengths,params.field_array,
                                                 params.imaging_field,params.spw_index_array,
						 params.channel_grid_indicies,
						 params.enabled_channels);
                printf(" <DONE>\n");
            }
            gridding_timer.stop();
        });
    }
    void grid_duel_pol(gridding_parameters & params) {
        gridding_barrier();
        gridding_future = std::async(std::launch::async, [&params] () {
	    gridding_timer.start();
            using namespace imaging;
            printf("GRIDDING DUEL POL...");
            typedef baseline_transform_policy<uvw_base_type, transform_disable_facet_rotation> baseline_transform_policy_type;
            typedef phase_transform_policy<visibility_base_type,
                    uvw_base_type,
                    transform_disable_phase_rotation> phase_transform_policy_type;
            typedef polarization_gridding_policy<visibility_base_type, uvw_base_type,
                    visibility_weights_base_type, convolution_base_type, grid_base_type,
                    phase_transform_policy_type, gridding_double_pol> polarization_gridding_policy_type;
            typedef convolution_policy<convolution_base_type,uvw_base_type,grid_base_type,
                    polarization_gridding_policy_type, convolution_precomputed_fir> convolution_policy_type;

            baseline_transform_policy_type uvw_transform; //standard: no uvw rotation
            phase_transform_policy_type phase_transform; //standard: no phase rotation

            polarization_gridding_policy_type polarization_policy(phase_transform,
                    params.output_buffer,
                    params.visibilities,
                    params.visibility_weights,
                    params.flags,
                    params.number_of_polarization_terms,
                    params.polarization_index,
                    params.second_polarization_index,
                    params.nx*params.ny,
                    params.channel_count);
            convolution_policy_type convolution_policy(params.nx,params.ny,2,params.conv_support,params.conv_oversample,
                    params.conv, polarization_policy);

            imaging::grid<visibility_base_type,uvw_base_type,
                    reference_wavelengths_base_type,convolution_base_type,
                    visibility_weights_base_type,grid_base_type,
                    baseline_transform_policy_type,
                    polarization_gridding_policy_type,
                    convolution_policy_type>
                    (polarization_policy,uvw_transform,convolution_policy,
                     params.uvw_coords,
                     params.flagged_rows,
                     params.nx,params.ny,
                     casa::Quantity(params.cell_size_x,"arcsec"),
                     casa::Quantity(params.cell_size_y,"arcsec"),
                     params.channel_count,
                     params.row_count,params.reference_wavelengths,params.field_array,
                     params.imaging_field,params.spw_index_array,
		     params.channel_grid_indicies,
		     params.enabled_channels);
	    gridding_timer.stop();
        });
    }

    void facet_duel_pol(gridding_parameters & params) {
        gridding_barrier();
        gridding_future = std::async(std::launch::async, [&params] () {
	    gridding_timer.start();
            using namespace imaging;
            size_t no_facet_pixels = params.nx*params.ny;
            for (size_t facet_index = 0; facet_index < params.num_facet_centres; ++facet_index) {
                uvw_base_type new_phase_ra = params.facet_centres[2*facet_index];
                uvw_base_type new_phase_dec = params.facet_centres[2*facet_index + 1];

                printf("FACETING (%f,%f,%f,%f) %lu / %lu...\n",params.phase_centre_ra,params.phase_centre_dec,new_phase_ra,new_phase_dec,facet_index+1, params.num_facet_centres);
                fflush(stdout);


                typedef imaging::baseline_transform_policy<uvw_base_type,
                        transform_facet_lefthanded_ra_dec> baseline_transform_policy_type;
                typedef imaging::phase_transform_policy<visibility_base_type,
                        uvw_base_type,
                        transform_enable_phase_rotation_lefthanded_ra_dec> phase_transform_policy_type;
                typedef imaging::polarization_gridding_policy<visibility_base_type, uvw_base_type,
                        visibility_weights_base_type, convolution_base_type, grid_base_type,
                        phase_transform_policy_type, gridding_double_pol> polarization_gridding_policy_type;
                typedef imaging::convolution_policy<convolution_base_type,uvw_base_type,grid_base_type,
                        polarization_gridding_policy_type,convolution_precomputed_fir> convolution_policy_type;
                baseline_transform_policy_type uvw_transform(0,0,casa::Quantity(params.phase_centre_ra,"arcsec"),casa::Quantity(params.phase_centre_dec,"arcsec"),
                        casa::Quantity(new_phase_ra,"arcsec"),casa::Quantity(new_phase_dec,"arcsec")); //lm faceting
                phase_transform_policy_type phase_transform(casa::Quantity(params.phase_centre_ra,"arcsec"),casa::Quantity(params.phase_centre_dec,"arcsec"),
                        casa::Quantity(new_phase_ra,"arcsec"),casa::Quantity(new_phase_dec,"arcsec")); //lm faceting

                polarization_gridding_policy_type polarization_policy(phase_transform,
                        params.output_buffer + no_facet_pixels*facet_index*2*params.cube_channel_dim_size,
                        params.visibilities,
                        params.visibility_weights,
                        params.flags,
                        params.number_of_polarization_terms,
                        params.polarization_index,
                        params.second_polarization_index,
                        params.nx*params.ny,
                        params.channel_count);
                convolution_policy_type convolution_policy(params.nx,params.ny,2,
                        params.conv_support,params.conv_oversample,
                        params.conv, polarization_policy);
                imaging::grid<visibility_base_type,uvw_base_type,
                        reference_wavelengths_base_type,convolution_base_type,
                        visibility_weights_base_type,grid_base_type,
                        baseline_transform_policy_type,
                        polarization_gridding_policy_type,
                        convolution_policy_type>(polarization_policy,uvw_transform,convolution_policy,
                                                 params.uvw_coords,
                                                 params.flagged_rows,
                                                 params.nx,params.ny,
                                                 casa::Quantity(params.cell_size_x,"arcsec"),casa::Quantity(params.cell_size_y,"arcsec"),
                                                 params.channel_count,
                                                 params.row_count,params.reference_wavelengths,params.field_array,
                                                 params.imaging_field,params.spw_index_array,
						 params.channel_grid_indicies,
						 params.enabled_channels);
            }
            gridding_timer.stop();
        });
    }
    void grid_4_cor(gridding_parameters & params) {
        gridding_barrier();
        gridding_future = std::async(std::launch::async, [&params] () {
	    gridding_timer.start();
            using namespace imaging;
            assert(params.number_of_polarization_terms == 4); //Only supports 4 correlation visibilties in this mode
            printf("GRIDDING...\n");
            typedef baseline_transform_policy<uvw_base_type, transform_disable_facet_rotation> baseline_transform_policy_type;
            typedef phase_transform_policy<visibility_base_type,
                    uvw_base_type,
                    transform_disable_phase_rotation> phase_transform_policy_type;
            typedef polarization_gridding_policy<visibility_base_type, uvw_base_type,
                    visibility_weights_base_type, convolution_base_type, grid_base_type,
                    phase_transform_policy_type, gridding_4_pol> polarization_gridding_policy_type;
            typedef convolution_policy<convolution_base_type,uvw_base_type,grid_base_type,
                    polarization_gridding_policy_type, convolution_precomputed_fir> convolution_policy_type;

            baseline_transform_policy_type uvw_transform; //standard: no uvw rotation
            phase_transform_policy_type phase_transform; //standard: no phase rotation

            polarization_gridding_policy_type polarization_policy(phase_transform,
                    params.output_buffer,
                    params.visibilities,
                    params.visibility_weights,
                    params.flags,params.nx*params.ny,
                    params.channel_count);
            convolution_policy_type convolution_policy(params.nx,params.ny,4,params.conv_support,params.conv_oversample,
                    params.conv, polarization_policy);

            imaging::grid<visibility_base_type,uvw_base_type,
                    reference_wavelengths_base_type,convolution_base_type,
                    visibility_weights_base_type,grid_base_type,
                    baseline_transform_policy_type,
                    polarization_gridding_policy_type,
                    convolution_policy_type>
                    (polarization_policy,uvw_transform,convolution_policy,
                     params.uvw_coords,
                     params.flagged_rows,
                     params.nx,params.ny,
                     casa::Quantity(params.cell_size_x,"arcsec"),
                     casa::Quantity(params.cell_size_y,"arcsec"),
                     params.channel_count,
                     params.row_count,params.reference_wavelengths,params.field_array,
                     params.imaging_field,params.spw_index_array,
		     params.channel_grid_indicies,
		     params.enabled_channels);
	    gridding_timer.stop();
        });
    }
    void facet_4_cor(gridding_parameters & params) {
        gridding_barrier();
        gridding_future = std::async(std::launch::async, [&params] () {
	    gridding_timer.start();
            using namespace imaging;
            assert(params.number_of_polarization_terms == 4); //Only supports 4 correlation visibilties in this mode

            size_t no_facet_pixels = params.nx*params.ny;
            for (size_t facet_index = 0; facet_index < params.num_facet_centres; ++facet_index) {
                uvw_base_type new_phase_ra = params.facet_centres[2*facet_index];
                uvw_base_type new_phase_dec = params.facet_centres[2*facet_index + 1];

                printf("FACETING (%f,%f,%f,%f) %lu / %lu...",params.phase_centre_ra,params.phase_centre_dec,new_phase_ra,new_phase_dec,facet_index+1, params.num_facet_centres);
                fflush(stdout);

                typedef imaging::baseline_transform_policy<uvw_base_type,
                        transform_facet_lefthanded_ra_dec> baseline_transform_policy_type;
                typedef imaging::phase_transform_policy<visibility_base_type,
                        uvw_base_type,
                        transform_enable_phase_rotation_lefthanded_ra_dec> phase_transform_policy_type;
                typedef imaging::polarization_gridding_policy<visibility_base_type, uvw_base_type,
                        visibility_weights_base_type, convolution_base_type, grid_base_type,
                        phase_transform_policy_type, gridding_4_pol> polarization_gridding_policy_type;
                typedef imaging::convolution_policy<convolution_base_type,uvw_base_type,grid_base_type,
                        polarization_gridding_policy_type,convolution_precomputed_fir> convolution_policy_type;

                baseline_transform_policy_type uvw_transform(0,0,casa::Quantity(params.phase_centre_ra,"arcsec"),casa::Quantity(params.phase_centre_dec,"arcsec"),
                        casa::Quantity(new_phase_ra,"arcsec"),casa::Quantity(new_phase_dec,"arcsec")); //lm faceting
                //baseline_transform_policy_type uvw_transform; //uv faceting
                phase_transform_policy_type phase_transform(casa::Quantity(params.phase_centre_ra,"arcsec"),casa::Quantity(params.phase_centre_dec,"arcsec"),
                        casa::Quantity(new_phase_ra,"arcsec"),casa::Quantity(new_phase_dec,"arcsec")); //lm faceting

                polarization_gridding_policy_type polarization_policy(phase_transform,
                        params.output_buffer + no_facet_pixels*facet_index*params.number_of_polarization_terms*params.cube_channel_dim_size,
                        params.visibilities,
                        params.visibility_weights,
                        params.flags,params.nx*params.ny,
                        params.channel_count);
                convolution_policy_type convolution_policy(params.nx,params.ny,4,
                        params.conv_support,params.conv_oversample,
                        params.conv, polarization_policy);
                imaging::grid<visibility_base_type,uvw_base_type,
                        reference_wavelengths_base_type,convolution_base_type,
                        visibility_weights_base_type,grid_base_type,
                        baseline_transform_policy_type,
                        polarization_gridding_policy_type,
                        convolution_policy_type>(polarization_policy,uvw_transform,convolution_policy,
                                                 params.uvw_coords,
                                                 params.flagged_rows,
                                                 params.nx,params.ny,
                                                 casa::Quantity(params.cell_size_x,"arcsec"),casa::Quantity(params.cell_size_y,"arcsec"),
                                                 params.channel_count,
                                                 params.row_count,params.reference_wavelengths,params.field_array,
                                                 params.imaging_field,params.spw_index_array,
						 params.channel_grid_indicies,
						 params.enabled_channels);
            }
            gridding_timer.stop();
        });
    }
    void facet_4_cor_corrections(gridding_parameters & params) {
        gridding_barrier();
        gridding_future = std::async(std::launch::async, [&params] () {
	    gridding_timer.start();
            using namespace imaging;
            assert(params.number_of_polarization_terms == 4); //Only supports 4 correlation visibilties in this mode

            size_t no_facet_pixels = params.nx*params.ny;
            if (params.should_invert_jones_terms)
                printf("INVERTING %zu JONES MATRICIES...\n",
                       params.no_timestamps_read*params.antenna_count*params.num_facet_centres*params.spw_count*params.channel_count);
            invert_all<visibility_base_type>((jones_2x2<visibility_base_type>*)(params.jones_terms),
                                             params.no_timestamps_read*params.antenna_count*params.num_facet_centres*params.spw_count*params.channel_count);
            for (size_t facet_index = 0; facet_index < params.num_facet_centres; ++facet_index) {
                uvw_base_type new_phase_ra = params.facet_centres[2*facet_index];
                uvw_base_type new_phase_dec = params.facet_centres[2*facet_index + 1];

                printf("FACETING (%f,%f,%f,%f) %lu / %lu...\n",params.phase_centre_ra,params.phase_centre_dec,new_phase_ra,new_phase_dec,facet_index+1, params.num_facet_centres);
                fflush(stdout);

                typedef imaging::baseline_transform_policy<uvw_base_type,
                        transform_facet_lefthanded_ra_dec> baseline_transform_policy_type;
                typedef imaging::phase_transform_policy<visibility_base_type,
                        uvw_base_type,
                        transform_enable_phase_rotation_lefthanded_ra_dec> phase_transform_policy_type;
                typedef imaging::polarization_gridding_policy<visibility_base_type, uvw_base_type,
                        visibility_weights_base_type, convolution_base_type, grid_base_type,
                        phase_transform_policy_type, gridding_4_pol_enable_facet_based_jones_corrections> polarization_gridding_policy_type;
                typedef imaging::convolution_policy<convolution_base_type,uvw_base_type,grid_base_type,
                        polarization_gridding_policy_type,convolution_precomputed_fir> convolution_policy_type;

                baseline_transform_policy_type uvw_transform(0,0,casa::Quantity(params.phase_centre_ra,"arcsec"),casa::Quantity(params.phase_centre_dec,"arcsec"),
                        casa::Quantity(new_phase_ra,"arcsec"),casa::Quantity(new_phase_dec,"arcsec")); //lm faceting
                //baseline_transform_policy_type uvw_transform; //uv faceting
                phase_transform_policy_type phase_transform(casa::Quantity(params.phase_centre_ra,"arcsec"),casa::Quantity(params.phase_centre_dec,"arcsec"),
                        casa::Quantity(new_phase_ra,"arcsec"),casa::Quantity(new_phase_dec,"arcsec")); //lm faceting
                polarization_gridding_policy_type polarization_policy(phase_transform,
                        params.output_buffer + no_facet_pixels*facet_index*params.number_of_polarization_terms*params.cube_channel_dim_size,
                        params.visibilities,
                        params.visibility_weights,
                        params.flags,params.nx*params.ny,
                        params.channel_count,
                        (jones_2x2<visibility_base_type>*)(params.jones_terms),
                        params.antenna_1_ids,
                        params.antenna_2_ids,
                        params.timestamp_ids,
                        params.antenna_count,
                        facet_index,
                        params.num_facet_centres,
                        params.no_timestamps_read,
                        params.spw_count
                                                                     );
                convolution_policy_type convolution_policy(params.nx,params.ny,4,
                        params.conv_support,params.conv_oversample,
                        params.conv, polarization_policy);
                imaging::grid<visibility_base_type,uvw_base_type,
                        reference_wavelengths_base_type,convolution_base_type,
                        visibility_weights_base_type,grid_base_type,
                        baseline_transform_policy_type,
                        polarization_gridding_policy_type,
                        convolution_policy_type>(polarization_policy,uvw_transform,convolution_policy,
                                                 params.uvw_coords,
                                                 params.flagged_rows,
                                                 params.nx,params.ny,
                                                 casa::Quantity(params.cell_size_x,"arcsec"),casa::Quantity(params.cell_size_y,"arcsec"),
                                                 params.channel_count,
                                                 params.row_count,params.reference_wavelengths,params.field_array,
                                                 params.imaging_field,params.spw_index_array,
						 params.channel_grid_indicies,
						 params.enabled_channels);
            }
        });
    }
    
    void grid_sampling_function(gridding_parameters & params) {
        gridding_barrier();
        gridding_future = std::async(std::launch::async, [&params] () {
	    gridding_timer.start();
            using namespace imaging;
            printf("GRIDDING SAMPLING FUNCTION...\n");
            typedef baseline_transform_policy<uvw_base_type, transform_disable_facet_rotation> baseline_transform_policy_type;
            typedef phase_transform_policy<visibility_base_type,
                    uvw_base_type,
                    transform_disable_phase_rotation> phase_transform_policy_type;
            typedef polarization_gridding_policy<visibility_base_type, uvw_base_type,
                    visibility_weights_base_type, convolution_base_type, grid_base_type,
                    phase_transform_policy_type, gridding_sampling_function> polarization_gridding_policy_type;
            typedef convolution_policy<convolution_base_type,uvw_base_type,grid_base_type,
                    polarization_gridding_policy_type, convolution_precomputed_fir> convolution_policy_type;

            baseline_transform_policy_type uvw_transform; //standard: no uvw rotation
            phase_transform_policy_type phase_transform; //standard: no phase rotation
            polarization_gridding_policy_type polarization_policy(phase_transform,
                    params.sampling_function_buffer,
                    params.flags,
		    params.visibility_weights,
                    params.number_of_polarization_terms,
                    0,
                    params.channel_count);
            convolution_policy_type convolution_policy(params.nx,params.ny,1,params.conv_support,params.conv_oversample,
                    params.conv, polarization_policy);

            imaging::grid<visibility_base_type,uvw_base_type,
                    reference_wavelengths_base_type,convolution_base_type,
                    visibility_weights_base_type,grid_base_type,
                    baseline_transform_policy_type,
                    polarization_gridding_policy_type,
                    convolution_policy_type>
                    (polarization_policy,uvw_transform,convolution_policy,
                     params.uvw_coords,
                     params.flagged_rows,
                     params.nx,params.ny,
                     casa::Quantity(params.cell_size_x,"arcsec"),
                     casa::Quantity(params.cell_size_y,"arcsec"),
                     params.channel_count,
                     params.row_count,params.reference_wavelengths,params.field_array,
                     params.imaging_field,params.spw_index_array,
		     params.sampling_function_channel_grid_indicies,
		     params.enabled_channels);
        });
    }
    
    void facet_sampling_function(gridding_parameters & params) {
        gridding_barrier();
        gridding_future = std::async(std::launch::async, [&params] () {
	    gridding_timer.start();
            using namespace imaging;
            size_t no_facet_pixels = params.nx*params.ny;
            for (size_t facet_index = 0; facet_index < params.num_facet_centres; ++facet_index) {
                uvw_base_type new_phase_ra = params.facet_centres[2*facet_index];
                uvw_base_type new_phase_dec = params.facet_centres[2*facet_index + 1];

                printf("FACETING SAMPLING FUNCTION (%f,%f,%f,%f) %lu / %lu...\n",params.phase_centre_ra,params.phase_centre_dec,new_phase_ra,new_phase_dec,facet_index+1, params.num_facet_centres);
                fflush(stdout);


                typedef imaging::baseline_transform_policy<uvw_base_type,
                        transform_facet_lefthanded_ra_dec> baseline_transform_policy_type;
                typedef imaging::phase_transform_policy<visibility_base_type,
                        uvw_base_type,
                        transform_enable_phase_rotation_lefthanded_ra_dec> phase_transform_policy_type;
                typedef imaging::polarization_gridding_policy<visibility_base_type, uvw_base_type,
                        visibility_weights_base_type, convolution_base_type, grid_base_type,
                        phase_transform_policy_type, gridding_sampling_function> polarization_gridding_policy_type;
                typedef imaging::convolution_policy<convolution_base_type,uvw_base_type,grid_base_type,
                        polarization_gridding_policy_type,convolution_precomputed_fir> convolution_policy_type;
                baseline_transform_policy_type uvw_transform(0,0,casa::Quantity(params.phase_centre_ra,"arcsec"),casa::Quantity(params.phase_centre_dec,"arcsec"),
                        casa::Quantity(new_phase_ra,"arcsec"),casa::Quantity(new_phase_dec,"arcsec")); //lm faceting
                phase_transform_policy_type phase_transform(casa::Quantity(params.phase_centre_ra,"arcsec"),casa::Quantity(params.phase_centre_dec,"arcsec"),
                        casa::Quantity(new_phase_ra,"arcsec"),casa::Quantity(new_phase_dec,"arcsec")); //lm faceting

                polarization_gridding_policy_type polarization_policy(phase_transform,
                        params.sampling_function_buffer + no_facet_pixels*params.sampling_function_channel_count*facet_index,
                        params.flags,
			params.visibility_weights,
                        params.number_of_polarization_terms,
                        0,
                        params.channel_count);
                convolution_policy_type convolution_policy(params.nx,params.ny,1,
                        params.conv_support,params.conv_oversample,
                        params.conv, polarization_policy);
                imaging::grid<visibility_base_type,uvw_base_type,
                        reference_wavelengths_base_type,convolution_base_type,
                        visibility_weights_base_type,grid_base_type,
                        baseline_transform_policy_type,
                        polarization_gridding_policy_type,
                        convolution_policy_type>(polarization_policy,uvw_transform,convolution_policy,
                                                 params.uvw_coords,
                                                 params.flagged_rows,
                                                 params.nx,params.ny,
                                                 casa::Quantity(params.cell_size_x,"arcsec"),casa::Quantity(params.cell_size_y,"arcsec"),
                                                 params.channel_count,
                                                 params.row_count,params.reference_wavelengths,params.field_array,
                                                 params.imaging_field,params.spw_index_array,
						 params.sampling_function_channel_grid_indicies,
						 params.enabled_channels);
                printf(" <DONE>\n");
            }
        });
    }
}
