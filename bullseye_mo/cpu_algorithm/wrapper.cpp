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
#include "templated_gridder.h"
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
	     "Backend: Alternative Multithreaded CPU Library\n\n");
      #ifdef BULLSEYE_SINGLE
      printf(" >Double precision mode: disabled \n");
      #endif
      #ifdef BULLSEYE_DOUBLE
      printf(" >Double precision mode: enabled \n");
      #endif
      #ifdef __AVX__
      #pragma message("Enabling AVX in compilation")
      printf(" >AVX Vectorization for w-projection modes: enabled\n");
      #else
      printf(" >AVX Vectorization for w-projection modes: disabled\n");
      #endif
      printf(" >Number of cores available: %d\n",omp_get_num_procs());
      printf(" >Number of threads being used: %d\n",omp_get_max_threads());
      printf("-----------------------------------------------\n");
      fftw_ifft_machine = new imaging::ifft_machine(params);
      sample_count_per_grid = new normalization_base_type[params.num_facet_centres * 
							  params.cube_channel_dim_size * 
							  params.number_of_polarization_terms_being_gridded]();
      params.normalization_terms = sample_count_per_grid;
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
	      normalization_base_type norm_val = params.normalization_terms[(f * params.cube_channel_dim_size + ch) * 
									    params.number_of_polarization_terms_being_gridded +
									    corr];
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
            {
	      printf("Gridding single correlation on the CPU...\n");
	      //invoke computation
	      {
		typedef imaging::correlation_gridding_policy<imaging::grid_single_correlation> correlation_gridding_policy;
		typedef imaging::baseline_transform_policy<imaging::transform_disable_facet_rotation > baseline_transform_policy;
		typedef imaging::phase_transform_policy<imaging::disable_faceting_phase_shift > phase_transform_policy;
		if (params.wplanes <= 1){
		  typedef imaging::convolution_policy<correlation_gridding_policy,imaging::convolution_AA_1D_precomputed> convolution_policy;
		  imaging::templated_gridder<correlation_gridding_policy,baseline_transform_policy,phase_transform_policy,convolution_policy>(params);
		} else {
		  #ifdef __AVX__
		  typedef imaging::convolution_policy<correlation_gridding_policy,imaging::convolution_w_projection_precomputed> convolution_policy;
		  imaging::templated_gridder<correlation_gridding_policy,baseline_transform_policy,phase_transform_policy,convolution_policy>(params);
		  #else
		  typedef imaging::convolution_policy<correlation_gridding_policy,imaging::convolution_w_projection_precomputed> convolution_policy;
		  imaging::templated_gridder<correlation_gridding_policy,baseline_transform_policy,phase_transform_policy,convolution_policy>(params);
		  #endif
		}
	      }
	    }
	    gridding_timer.stop();
        });
    }
    void facet_single_pol(gridding_parameters & params) {
        gridding_future = std::async(std::launch::async, [&params] () {
	    gridding_timer.start();
	    {
	      printf("Faceting single correlation on the CPU...\n");    
	      //invoke computation
	      {
		typedef imaging::correlation_gridding_policy<imaging::grid_single_correlation> correlation_gridding_policy;
		typedef imaging::baseline_transform_policy<imaging::transform_planar_approx_with_w > baseline_transform_policy;
		typedef imaging::phase_transform_policy<imaging::enable_faceting_phase_shift> phase_transform_policy;
		if (params.wplanes <= 1){
		  typedef imaging::convolution_policy<correlation_gridding_policy,imaging::convolution_AA_1D_precomputed> convolution_policy;
		  imaging::templated_gridder<correlation_gridding_policy,baseline_transform_policy,phase_transform_policy,convolution_policy>(params);
		} else {
		  #ifdef __AVX__
		  typedef imaging::convolution_policy<correlation_gridding_policy,imaging::convolution_w_projection_precomputed_vectorized> convolution_policy;
		  imaging::templated_gridder<correlation_gridding_policy,baseline_transform_policy,phase_transform_policy,convolution_policy>(params);
		  #else
		  typedef imaging::convolution_policy<correlation_gridding_policy,imaging::convolution_w_projection_precomputed> convolution_policy;
		  imaging::templated_gridder<correlation_gridding_policy,baseline_transform_policy,phase_transform_policy,convolution_policy>(params);
		  #endif
		}
	      }
	    }
            gridding_timer.stop();
        });
    }
    void grid_duel_pol(gridding_parameters & params) {
        gridding_future = std::async(std::launch::async, [&params] () {
	    gridding_timer.start();
	    printf("Gridding duel correlation on the CPU...\n");  
	    typedef imaging::correlation_gridding_policy<imaging::grid_duel_correlation> correlation_gridding_policy;
	    typedef imaging::baseline_transform_policy<imaging::transform_disable_facet_rotation > baseline_transform_policy;
	    typedef imaging::phase_transform_policy<imaging::disable_faceting_phase_shift > phase_transform_policy;
	    if (params.wplanes <= 1){
	      typedef imaging::convolution_policy<correlation_gridding_policy,imaging::convolution_AA_1D_precomputed> convolution_policy;
	      imaging::templated_gridder<correlation_gridding_policy,baseline_transform_policy,phase_transform_policy,convolution_policy>(params);
	    } else {
	      #ifdef __AVX__
	      typedef imaging::convolution_policy<correlation_gridding_policy,imaging::convolution_w_projection_precomputed_vectorized> convolution_policy;
	      imaging::templated_gridder<correlation_gridding_policy,baseline_transform_policy,phase_transform_policy,convolution_policy>(params);
	      #else
	      typedef imaging::convolution_policy<correlation_gridding_policy,imaging::convolution_w_projection_precomputed> convolution_policy;
	      imaging::templated_gridder<correlation_gridding_policy,baseline_transform_policy,phase_transform_policy,convolution_policy>(params);
	      #endif
	    }
	    gridding_timer.stop();
        });
    }

    void facet_duel_pol(gridding_parameters & params) {
        gridding_future = std::async(std::launch::async, [&params] () {
	gridding_timer.start();
	printf("Faceting duel correlation on the CPU...\n");  
	{
	  typedef imaging::correlation_gridding_policy<imaging::grid_duel_correlation> correlation_gridding_policy;
	  typedef imaging::baseline_transform_policy<imaging::transform_planar_approx_with_w > baseline_transform_policy;
	  typedef imaging::phase_transform_policy<imaging::enable_faceting_phase_shift> phase_transform_policy;
	  if (params.wplanes <= 1){
	    typedef imaging::convolution_policy<correlation_gridding_policy,imaging::convolution_AA_1D_precomputed> convolution_policy;
	    imaging::templated_gridder<correlation_gridding_policy,baseline_transform_policy,phase_transform_policy,convolution_policy>(params);
	  } else {
	    #ifdef __AVX__
	    typedef imaging::convolution_policy<correlation_gridding_policy,imaging::convolution_w_projection_precomputed_vectorized> convolution_policy;
	    imaging::templated_gridder<correlation_gridding_policy,baseline_transform_policy,phase_transform_policy,convolution_policy>(params);
	    #else
	    typedef imaging::convolution_policy<correlation_gridding_policy,imaging::convolution_w_projection_precomputed> convolution_policy;
	    imaging::templated_gridder<correlation_gridding_policy,baseline_transform_policy,phase_transform_policy,convolution_policy>(params);
	    #endif
	  }
	}
            gridding_timer.stop();
        });
    }
    void grid_4_cor(gridding_parameters & params) {
        gridding_future = std::async(std::launch::async, [&params] () {
	    gridding_timer.start();
	    printf("Gridding quad correlation on the CPU...\n");  
	    typedef imaging::correlation_gridding_policy<imaging::grid_4_correlation> correlation_gridding_policy;
	    typedef imaging::baseline_transform_policy<imaging::transform_disable_facet_rotation > baseline_transform_policy;
	    typedef imaging::phase_transform_policy<imaging::disable_faceting_phase_shift > phase_transform_policy;
	    if (params.wplanes <= 1){
	      typedef imaging::convolution_policy<correlation_gridding_policy,imaging::convolution_AA_1D_precomputed> convolution_policy;
	      imaging::templated_gridder<correlation_gridding_policy,baseline_transform_policy,phase_transform_policy,convolution_policy>(params);
	    } else {
	      #ifdef __AVX__
	      typedef imaging::convolution_policy<correlation_gridding_policy,imaging::convolution_w_projection_precomputed_vectorized> convolution_policy;
	      imaging::templated_gridder<correlation_gridding_policy,baseline_transform_policy,phase_transform_policy,convolution_policy>(params);
	      #else
	      typedef imaging::convolution_policy<correlation_gridding_policy,imaging::convolution_w_projection_precomputed> convolution_policy;
	      imaging::templated_gridder<correlation_gridding_policy,baseline_transform_policy,phase_transform_policy,convolution_policy>(params);
	      #endif
	    }
	    gridding_timer.stop();
        });
    }
    void facet_4_cor(gridding_parameters & params) {
        gridding_future = std::async(std::launch::async, [&params] () {
	    gridding_timer.start();
            printf("Faceting quad correlation on the CPU...\n");  
	    typedef imaging::correlation_gridding_policy<imaging::grid_4_correlation> correlation_gridding_policy;
	    typedef imaging::baseline_transform_policy<imaging::transform_planar_approx_with_w > baseline_transform_policy;
	    typedef imaging::phase_transform_policy<imaging::enable_faceting_phase_shift> phase_transform_policy;
	    if (params.wplanes <= 1){
	      typedef imaging::convolution_policy<correlation_gridding_policy,imaging::convolution_AA_1D_precomputed> convolution_policy;
	      imaging::templated_gridder<correlation_gridding_policy,baseline_transform_policy,phase_transform_policy,convolution_policy>(params);
	    } else {
	      #ifdef __AVX__
	      typedef imaging::convolution_policy<correlation_gridding_policy,imaging::convolution_w_projection_precomputed_vectorized> convolution_policy;
	      imaging::templated_gridder<correlation_gridding_policy,baseline_transform_policy,phase_transform_policy,convolution_policy>(params);
	      #else
	      typedef imaging::convolution_policy<correlation_gridding_policy,imaging::convolution_w_projection_precomputed> convolution_policy;
	      imaging::templated_gridder<correlation_gridding_policy,baseline_transform_policy,phase_transform_policy,convolution_policy>(params);
	      #endif
	    }
            gridding_timer.stop();
        });
    }
    void facet_4_cor_corrections(gridding_parameters & params) {
        gridding_future = std::async(std::launch::async, [&params] () {
	    gridding_timer.start();
	    printf("Faceting with jones corrections on the CPU...\n");
	    std::size_t no_terms_to_invert = params.no_timestamps_read*params.antenna_count*
						   params.num_facet_centres*params.spw_count*
						   params.channel_count;
	    printf("---Inverting %lu jones matricies before gridding operation...\n",no_terms_to_invert);
	    imaging::invert_all((imaging::jones_2x2<visibility_base_type> *)params.jones_terms,no_terms_to_invert);
            typedef imaging::correlation_gridding_policy<imaging::grid_4_correlation_with_jones_corrections> correlation_gridding_policy;
	    typedef imaging::baseline_transform_policy<imaging::transform_planar_approx_with_w > baseline_transform_policy;
	    typedef imaging::phase_transform_policy<imaging::enable_faceting_phase_shift> phase_transform_policy;
	    if (params.wplanes <= 1){
	      typedef imaging::convolution_policy<correlation_gridding_policy,imaging::convolution_AA_1D_precomputed> convolution_policy;
	      imaging::templated_gridder<correlation_gridding_policy,baseline_transform_policy,phase_transform_policy,convolution_policy>(params);
	    } else {
	      #ifdef __AVX__
	      typedef imaging::convolution_policy<correlation_gridding_policy,imaging::convolution_w_projection_precomputed_vectorized> convolution_policy;
	      imaging::templated_gridder<correlation_gridding_policy,baseline_transform_policy,phase_transform_policy,convolution_policy>(params);
	      #else
	      typedef imaging::convolution_policy<correlation_gridding_policy,imaging::convolution_w_projection_precomputed> convolution_policy;
	      imaging::templated_gridder<correlation_gridding_policy,baseline_transform_policy,phase_transform_policy,convolution_policy>(params);
	      #endif
	    }
            gridding_timer.stop();
        });
    }
    
    void grid_sampling_function(gridding_parameters & params) {
        gridding_future = std::async(std::launch::async, [&params] () {
	    sampling_function_gridding_timer.start();
            printf("Gridding sampling function on the CPU...\n");  
            typedef imaging::correlation_gridding_policy<imaging::grid_sampling_function> correlation_gridding_policy;
	    typedef imaging::baseline_transform_policy<imaging::transform_disable_facet_rotation > baseline_transform_policy;
	    typedef imaging::phase_transform_policy<imaging::disable_faceting_phase_shift > phase_transform_policy;
	    if (params.wplanes <= 1){
	      typedef imaging::convolution_policy<correlation_gridding_policy,imaging::convolution_AA_1D_precomputed> convolution_policy;
	      imaging::templated_gridder<correlation_gridding_policy,baseline_transform_policy,phase_transform_policy,convolution_policy>(params);
	    } else {
	      typedef imaging::convolution_policy<correlation_gridding_policy,imaging::convolution_w_projection_precomputed> convolution_policy;
	      imaging::templated_gridder<correlation_gridding_policy,baseline_transform_policy,phase_transform_policy,convolution_policy>(params);
	    }
	    sampling_function_gridding_timer.stop();
        });
    }
    
    void facet_sampling_function(gridding_parameters & params) {
        gridding_future = std::async(std::launch::async, [&params] () {
	    sampling_function_gridding_timer.start();
            printf("Faceting sampling function on the CPU...\n");
	    typedef imaging::correlation_gridding_policy<imaging::grid_sampling_function> correlation_gridding_policy;
	    typedef imaging::baseline_transform_policy<imaging::transform_planar_approx_with_w > baseline_transform_policy;
	    typedef imaging::phase_transform_policy<imaging::disable_faceting_phase_shift > phase_transform_policy;
	    if (params.wplanes <= 1){
	      typedef imaging::convolution_policy<correlation_gridding_policy,imaging::convolution_AA_1D_precomputed> convolution_policy;
	      imaging::templated_gridder<correlation_gridding_policy,baseline_transform_policy,phase_transform_policy,convolution_policy>(params);
	    } else {
	      typedef imaging::convolution_policy<correlation_gridding_policy,imaging::convolution_w_projection_precomputed> convolution_policy;
	      imaging::templated_gridder<correlation_gridding_policy,baseline_transform_policy,phase_transform_policy,convolution_policy>(params);
	    }
            sampling_function_gridding_timer.stop();
        });
    }
}
