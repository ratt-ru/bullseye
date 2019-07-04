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
#pragma once
#include "cu_common.h"
#include "gridding_parameters.h"
#include "cu_basic_complex.h"

namespace imaging {
  /**
   * This is a basic implementation of a gpu-base "Direct" FT
   * It expects to be launched with (1 thread per pixel):
   * no_blocks_per_grid.x * no_threads_per_block.x = nx
   * no_blocks_per_grid.y * no_threads_per_block.y = ny
   */
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

    for (std::size_t bt = 0; bt < params.row_count; ++bt){
	imaging::uvw_coord<uvw_base_type> uvw = params.uvw_coords[bt];
	unsigned int spw_offset = params.spw_index_array[bt] * params.channel_count;
	bool row_flag = params.flagged_rows[bt];
	size_t row_offset = bt*params.channel_count;
	for (std::size_t c = 0; c < params.channel_count; ++c){
	    visibility_weights_base_type weight = params.visibility_weights[row_offset + c];
	    visibility_base_type channel_enabled_filter = (visibility_base_type)(int)(!(row_flag || params.flags[row_offset + spw_offset + c]) &&
											params.enabled_channels[spw_offset + c]);
	    
	    reference_wavelengths_base_type reciprocal_wavelength = 1 / params.reference_wavelengths[spw_offset + c];
	    float exp_arg = 2*M_PI*(uvw._u * reciprocal_wavelength * l + uvw._v * reciprocal_wavelength * m);
	    basic_complex<visibility_base_type> vis = ((basic_complex<visibility_base_type>*)params.visibilities)[bt*params.channel_count + c];
	    vis._real *= weight * channel_enabled_filter;
	    vis._imag *= weight * channel_enabled_filter;
	    //by Euler's identity
	    accum_grid_val += vis._real * cos(exp_arg) - vis._imag * sin(exp_arg); 
	}
    }
    ((grid_base_type*)params.output_buffer)[grid_flat_index] = accum_grid_val;
  }
}
