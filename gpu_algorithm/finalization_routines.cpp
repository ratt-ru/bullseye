#include <fftw3.h>
#include <stdexcept>
#include "wrapper.h"
#include "fft_shift_utils.h"
#include "timer.h"
#ifdef SHOULD_DO_32_BIT_FFT
typedef fftwf_plan fftw_plan_type;
#else
typedef fftw_plan fftw_plan_type;
#endif

extern "C" {
    double total_inversion_time = 0;
    double get_inversion_walltime(){
      return total_inversion_time;
    }
    void normalize(gridding_parameters & params) {
      printf("WARNING: GPU NORMALIZATION NOT IMPLEMENTED\n");
    }
    void finalize(gridding_parameters & params) {
        gridding_barrier();
	cudaStream_t inversion_timing_stream;
	cudaSafeCall(cudaStreamCreateWithFlags(&inversion_timing_stream,cudaStreamNonBlocking));
	utils::timer inversion_walltime(inversion_timing_stream);
	inversion_walltime.start();
        int dims[] = {(int)params.nx,(int)params.ny};
        fftw_plan_type fft_plan;
	
	#ifdef SHOULD_DO_32_BIT_FFT
        fft_plan = fftwf_plan_many_dft(2,(int*)&dims,
                                       params.cube_channel_dim_size,
                                       (fftwf_complex *)params.output_buffer,(int*)&dims,
                                       1,(int)(params.nx*params.ny),
                                       (fftwf_complex *)params.output_buffer,(int*)&dims,
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
	#endif
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
		  grid_ptr_single[i] = (float)(grid_ptr_gridtype[i*2]); //extract all the reals
	      }
	    }
        }

	#ifdef SHOULD_DO_32_BIT_FFT
        fftwf_destroy_plan(fft_plan);
	#else
        fftw_destroy_plan(fft_plan);
	#endif
	inversion_walltime.stop();
	cudaSafeCall(cudaStreamDestroy(inversion_timing_stream));
	total_inversion_time += inversion_walltime.duration();
    }
    void finalize_psf(gridding_parameters & params) {
        gridding_barrier();
	cudaStream_t inversion_timing_stream;
	cudaSafeCall(cudaStreamCreateWithFlags(&inversion_timing_stream,cudaStreamNonBlocking));
	utils::timer inversion_walltime(inversion_timing_stream);
	inversion_walltime.start();
	fftw_plan_type fft_psf_plan;
	int dims[] = {(int)params.nx,(int)params.ny};
	#ifdef SHOULD_DO_32_BIT_FFT
	fft_psf_plan = fftwf_plan_many_dft(2,(int*)&dims,
					   params.sampling_function_channel_count * params.num_facet_centres,
					   (fftwf_complex *)params.sampling_function_buffer,(int*)&dims,
					   1,(int)(params.nx*params.ny),
					   (fftwf_complex *)params.sampling_function_buffer,(int*)&dims,
					   1,(int)(params.nx*params.ny),
					   FFTW_BACKWARD,FFTW_ESTIMATE | FFTW_UNALIGNED);
	#else
	fft_psf_plan = fftw_plan_many_dft(2,(int*)&dims,
					  params.sampling_function_channel_count * params.num_facet_centres,
					  (fftw_complex *)params.sampling_function_buffer,(int*)&dims,
					  1,(int)(params.nx*params.ny),
					  (fftw_complex *)params.sampling_function_buffer,(int*)&dims,
					  1,(int)(params.nx*params.ny),
					  FFTW_BACKWARD,FFTW_ESTIMATE | FFTW_UNALIGNED);
	#endif
	std::size_t offset = params.nx*params.ny*params.sampling_function_channel_count;
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
		  grid_ptr_single[i] = (float)(grid_ptr_gridtype[i*2]); //extract all the reals
	      }
	  }
	}
	#ifdef SHOULD_DO_32_BIT_FFT
        fftwf_destroy_plan(fft_psf_plan);
	#else
        fftw_destroy_plan(fft_psf_plan);
	#endif
	inversion_walltime.stop();
	cudaSafeCall(cudaStreamDestroy(inversion_timing_stream));
	total_inversion_time += inversion_walltime.duration();
    }
}