#include <fftw3.h>
#include <stdexcept>

#include "wrapper.h"
#include "fft_shift_utils.h"

#ifdef SHOULD_DO_32_BIT_FFT
typedef fftwf_plan fftw_plan_type;
#else
typedef fftw_plan fftw_plan_type;
#endif

extern "C" {
    void finalize(gridding_parameters & params) {
        gridding_barrier();
        int dims[] = {(int)params.nx,(int)params.ny};
        fftw_plan_type fft_plan;
        fftw_plan_type fft_psf_plan;
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

//       inversion_walltime->start();
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
//       inversion_walltime->stop();
#ifdef SHOULD_DO_32_BIT_FFT
        fftwf_destroy_plan(fft_plan);
#else
        fftw_destroy_plan(fft_plan);
#endif
    }
    void finalize_psf(gridding_parameters & params) {
        throw std::runtime_error("Backend Unimplemented exception: finalize_psf");
    }
}