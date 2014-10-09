#pragma once
#include "gridding_parameters.h"
namespace imaging {
  template <typename T> struct basic_complex { T _real,_imag; };
  const float ARCSEC_TO_RAD = M_PI/(180.0*3600.0);
  __global__ void dft_kernel(gridding_parameters params,
			     dim3 no_blocks_per_grid, dim3 no_threads_per_block){
    //the l,m coordinate is based on the offsetted (x,y) thread index
    size_t x = threadIdx.x + blockIdx.x*blockDim.x;
    size_t y = threadIdx.y + blockIdx.y*blockDim.y;
    size_t grid_flat_index = x + y * params.nx;
    grid_base_type accum_grid_val = ((grid_base_type *)params.output_buffer)[grid_flat_index];
    uvw_base_type ra = (x - params.nx/2.0f) * params.cell_size_x * ARCSEC_TO_RAD;
    uvw_base_type dec = (y - params.ny/2.0f) * params.cell_size_y * ARCSEC_TO_RAD;
    uvw_base_type l = cosf(dec)*sinf(ra);
    uvw_base_type m = -sinf(dec);
    uvw_base_type n = sqrtf(1 - l*l - m*m);
    for (std::size_t bt = 0; bt < params.row_count; ++bt){
	imaging::uvw_coord<uvw_base_type> uvw = params.uvw_coords[bt];
#pragma unroll 8
	for (std::size_t c = 0; c < params.channel_count; ++c){
	    basic_complex<visibility_base_type> vis = ((basic_complex<visibility_base_type>*)params.visibilities)[bt*params.channel_count + c];       
	    float exp_arg = 2*M_PI*(uvw._u/0.208009 * l + uvw._v/0.208009 * m + uvw._w/0.208009*(n-1));
	    basic_complex<uvw_base_type> phase_term = {cosf(exp_arg),sinf(exp_arg)}; 
	    //by Euler's identity
	    accum_grid_val += vis._real * phase_term._real - vis._imag * phase_term._imag; 
	}
    }
    ((grid_base_type*)params.output_buffer)[grid_flat_index] = accum_grid_val;
  }
}
