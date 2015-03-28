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
#include "fft_and_repacking_routines.h"
#include <fftw3.h>
#ifdef SHOULD_DO_32_BIT_FFT
typedef fftwf_plan fftw_plan_type;
#else
typedef fftw_plan fftw_plan_type;
#endif
namespace imaging{
  ifft_machine::ifft_machine(gridding_parameters & params){
    int dims[] = {(int)params.nx,(int)params.ny};
      fft_plan = (void*) new fftw_plan_type;
      fft_psf_plan = (void*) new fftw_plan_type;
      #ifdef SHOULD_DO_32_BIT_FFT
	*((fftw_plan_type *)fft_plan) = fftwf_plan_many_dft(2,(int*)&dims,
				       params.cube_channel_dim_size,
				       (fftwf_complex *)params.output_buffer,(int*)&dims,
				       1,(int)(params.nx*params.ny),
				       (fftwf_complex *)params.output_buffer,(int*)&dims,
				       1,(int)(params.nx*params.ny),
				       FFTW_BACKWARD,FFTW_ESTIMATE | FFTW_UNALIGNED);
	*((fftw_plan_type *)fft_psf_plan) = fftwf_plan_many_dft(2,(int*)&dims,
					   params.sampling_function_channel_count * params.num_facet_centres,
					   (fftwf_complex *)params.sampling_function_buffer,(int*)&dims,
					   1,(int)(params.nx*params.ny),
					   (fftwf_complex *)params.sampling_function_buffer,(int*)&dims,
					   1,(int)(params.nx*params.ny),
					   FFTW_BACKWARD,FFTW_ESTIMATE | FFTW_UNALIGNED);
      #else
	*((fftw_plan_type *)fft_plan) = fftw_plan_many_dft(2,(int*)&dims,
				      params.cube_channel_dim_size,
				      (fftw_complex *)params.output_buffer,(int*)&dims,
				      1,(int)(params.nx*params.ny),
				      (fftw_complex *)params.output_buffer,(int*)&dims,
				      1,(int)(params.nx*params.ny),
				      FFTW_BACKWARD,FFTW_ESTIMATE | FFTW_UNALIGNED);
	*((fftw_plan_type *)fft_psf_plan) = fftw_plan_many_dft(2,(int*)&dims,
					  params.sampling_function_channel_count * params.num_facet_centres,
					  (fftw_complex *)params.sampling_function_buffer,(int*)&dims,
					  1,(int)(params.nx*params.ny),
					  (fftw_complex *)params.sampling_function_buffer,(int*)&dims,
					  1,(int)(params.nx*params.ny),
					  FFTW_BACKWARD,FFTW_ESTIMATE | FFTW_UNALIGNED);
      #endif
  }
  void ifft_machine::repack_and_ifft_uv_grids(gridding_parameters & params){
	std::size_t offset = params.nx*params.ny*params.cube_channel_dim_size*params.number_of_polarization_terms_being_gridded;
	#ifdef SHOULD_DO_32_BIT_FFT
	  for (std::size_t f = 0; f < params.num_facet_centres; ++f) {
	    utils::ifftshift(params.output_buffer + f*offset,params.nx,params.ny,params.cube_channel_dim_size);
	    fftwf_execute_dft(*((fftw_plan_type *)fft_plan),
			      (fftwf_complex *)(params.output_buffer + f*offset),
			      (fftwf_complex *)(params.output_buffer + f*offset));
	    utils::fftshift(params.output_buffer + f*offset,params.nx,params.ny,params.cube_channel_dim_size);
	  }
	#else
	  for (std::size_t f = 0; f < params.num_facet_centres; ++f) {
	    utils::ifftshift(params.output_buffer + f*offset,params.nx,params.ny,params.cube_channel_dim_size);
	    fftw_execute_dft(*((fftw_plan_type *)fft_plan),
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
		  grid_ptr_single[i] = (float)(grid_ptr_gridtype[i*2]); //extract all the reals
	      }
	  }
	}
  }
  void ifft_machine::repack_and_ifft_sampling_function_grids(gridding_parameters & params){
	std::size_t offset = params.nx*params.ny*params.sampling_function_channel_count;
	#ifdef SHOULD_DO_32_BIT_FFT
	  for (std::size_t f = 0; f < params.num_facet_centres; ++f) 
	    utils::ifftshift(params.sampling_function_buffer + f*offset,params.nx,params.ny,params.sampling_function_channel_count);
	  fftwf_execute(*((fftw_plan_type *)fft_psf_plan));
 	  for (std::size_t f = 0; f < params.num_facet_centres; ++f) 
	    utils::fftshift(params.sampling_function_buffer + f*offset,params.nx,params.ny,params.sampling_function_channel_count);
	#else
	  for (std::size_t f = 0; f < params.num_facet_centres; ++f) 
	    utils::ifftshift(params.sampling_function_buffer + f*offset,params.nx,params.ny,params.sampling_function_channel_count);
	  fftw_execute(*((fftw_plan_type *)fft_psf_plan));
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
		  grid_ptr_single[i] = (float)(grid_ptr_gridtype[i*2]); //extract all the reals
	      }
	  }
	}
  }
  ifft_machine::~ifft_machine(){
    #ifdef SHOULD_DO_32_BIT_FFT
	fftwf_destroy_plan(*((fftw_plan_type *)fft_plan));
	fftwf_destroy_plan(*((fftw_plan_type *)fft_psf_plan));
    #else
	fftw_destroy_plan(*((fftw_plan_type *)fft_plan));
	fftw_destroy_plan(*((fftw_plan_type *)fft_psf_plan));
    #endif
    delete (fftw_plan_type *)fft_plan;
    delete (fftw_plan_type *)fft_psf_plan;
  }
}