#pragma once
#include "gridding_parameters.h"
#include "fft_shift_utils.h"
namespace imaging{
    class ifft_machine {
    private:
      void* fft_plan; //bastardized workaround for a bug in the FFTW header... cant include the header from a .cu file
      void* fft_psf_plan;
    public:
      ifft_machine(gridding_parameters & params);
      void repack_and_ifft_uv_grids(gridding_parameters & params);
      void repack_and_ifft_sampling_function_grids(gridding_parameters & params);
      virtual ~ifft_machine();
    };
}