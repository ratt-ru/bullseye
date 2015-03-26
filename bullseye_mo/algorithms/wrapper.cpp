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
#include "fft_and_repacking_routines.h"

extern "C" {
    imaging::ifft_machine * fftw_ifft_machine;
    utils::timer gridding_timer;
    utils::timer sampling_function_gridding_timer;
    utils::timer inversion_timer;
    std::future<void> gridding_future;
    normalization_base_type * sample_count_per_grid;
    bool initialized = false;
    
    double get_gridding_walltime() {
      return gridding_timer.duration() + sampling_function_gridding_timer.duration();
    }
    double get_inversion_walltime() {
      return inversion_timer.duration();
    }
    void gridding_barrier() {
        if (gridding_future.valid())
            gridding_future.get(); //Block until result becomes available
    }
    void initLibrary(gridding_parameters & params){
      if (initialized) return;
      initialized = true;
      printf("-----------------------------------------------\n"
	     "      Backend: Multithreaded CPU Library       \n\n");
      #ifdef BULLSEYE_SINGLE
      printf(" >Double precision mode: disabled \n");
      #endif
      #ifdef BULLSEYE_DOUBLE
      printf(" >Double precision mode: enabled \n");
      #endif
      printf(" >Number of cores available: %d\n",omp_get_num_procs());
      printf(" >Number of threads being used: %d\n",omp_get_max_threads());
      printf("-----------------------------------------------\n");
      fftw_ifft_machine = new imaging::ifft_machine(params);
      sample_count_per_grid = new normalization_base_type[omp_get_max_threads() *
							  params.num_facet_centres * 
							  params.cube_channel_dim_size * 
							  params.number_of_polarization_terms_being_gridded]();
    }
    void releaseLibrary(){
      if (!initialized) return;
      initialized = false;
      gridding_barrier();
      delete fftw_ifft_machine;
      delete [] sample_count_per_grid;
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
		for (size_t corr = 0; corr < params.number_of_polarization_terms_being_gridded; ++corr)
		  params.output_buffer[(((f*params.cube_channel_dim_size + g)*params.number_of_polarization_terms_being_gridded+corr)*params.ny+y)*params.nx+x] *= count;
	    }
    }
    void normalize(gridding_parameters & params){
	gridding_barrier();
	/*
	 * Now normalize per facet, per channel accumulator grid and correlation
	 */
	for (size_t f = 0; f < params.num_facet_centres; ++f){
	  for (size_t ch = 0; ch < params.cube_channel_dim_size; ++ch){
	    for (size_t corr = 0; corr < params.number_of_polarization_terms_being_gridded; ++corr){
	      normalization_base_type norm_val = 0;
	      for (size_t thid = 0; thid < omp_get_max_threads(); ++thid)
		norm_val += sample_count_per_grid[((thid * params.num_facet_centres + f) * 
						   params.cube_channel_dim_size + ch) * 
						  params.number_of_polarization_terms_being_gridded + corr];
	      printf("Normalizing cube slice @ facet %lu, channel slice %lu, correlation term %lu with val %f\n",
		     f,ch,corr,norm_val);
	      std::complex<grid_base_type> * __restrict__ grid_ptr = params.output_buffer + 
								     ((f * params.cube_channel_dim_size + ch) * 
								     params.number_of_polarization_terms_being_gridded + corr) * params.ny * params.nx;
	      for (size_t i = 0; i < params.ny * params.nx; ++i)
		grid_ptr[i] /= norm_val;
	    }
	  }
	}
    }
    void finalize(gridding_parameters & params){
	gridding_barrier();
	inversion_timer.start();
	fftw_ifft_machine->repack_and_ifft_uv_grids(params);
	inversion_timer.stop();
    }
    void finalize_psf(gridding_parameters & params){
	gridding_barrier();
	inversion_timer.start();
	fftw_ifft_machine->repack_and_ifft_sampling_function_grids(params);
	inversion_timer.stop();
    }
    void repack_input_data(gridding_parameters & params){
      throw std::runtime_error("Unimplemented: CPU data repacking not necessary");
    }
    void grid_single_pol(gridding_parameters & params) {
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
                    params.channel_count,
		    sample_count_per_grid,
		    params.cube_channel_dim_size,
		    params.num_facet_centres);
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
		     params.enabled_channels,0);
	    gridding_timer.stop();
        });
    }
    void facet_single_pol(gridding_parameters & params) {
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
                        params.channel_count,
			sample_count_per_grid,
			params.cube_channel_dim_size,
			params.num_facet_centres);
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
						 params.enabled_channels,facet_index);
                printf(" <DONE>\n");
            }
            gridding_timer.stop();
        });
    }
    void grid_duel_pol(gridding_parameters & params) {
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
                    params.channel_count,
		    sample_count_per_grid,
		    params.cube_channel_dim_size,
		    params.num_facet_centres);
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
		     params.enabled_channels,0);
	    gridding_timer.stop();
        });
    }

    void facet_duel_pol(gridding_parameters & params) {
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
                        params.channel_count,
			sample_count_per_grid,
			params.cube_channel_dim_size,
			params.num_facet_centres);
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
						 params.enabled_channels,facet_index);
            }
            gridding_timer.stop();
        });
    }
    void grid_4_cor(gridding_parameters & params) {
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
                    params.channel_count,
		    sample_count_per_grid,
		    params.cube_channel_dim_size,
		    params.num_facet_centres);
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
		     params.enabled_channels,0);
	    gridding_timer.stop();
        });
    }
    void facet_4_cor(gridding_parameters & params) {
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
                        params.channel_count,
			sample_count_per_grid,
			params.cube_channel_dim_size,
			params.num_facet_centres);
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
						 params.enabled_channels,facet_index);
            }
            gridding_timer.stop();
        });
    }
    void facet_4_cor_corrections(gridding_parameters & params) {
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
                        params.spw_count,
			sample_count_per_grid,
			params.cube_channel_dim_size,
			params.num_facet_centres);
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
						 params.enabled_channels,facet_index);
            }
            gridding_timer.stop();
        });
    }
    
    void grid_sampling_function(gridding_parameters & params) {
        gridding_future = std::async(std::launch::async, [&params] () {
	    sampling_function_gridding_timer.start();
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
		     params.enabled_channels,0);
	    sampling_function_gridding_timer.stop();
        });
    }
    
    void facet_sampling_function(gridding_parameters & params) {
        gridding_future = std::async(std::launch::async, [&params] () {
	    sampling_function_gridding_timer.start();
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
						 params.enabled_channels,facet_index);
                printf(" <DONE>\n");
            }
            sampling_function_gridding_timer.stop();
        });
    }
}
