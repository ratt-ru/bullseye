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
#include "fft_shift_utils.h"
namespace utils{
  void fftshift(std::complex<grid_base_type> * __restrict__ grid,
              std::size_t nx, std::size_t ny,
              std::size_t no_slices) {
    for (std::size_t slice = 0; slice < no_slices; ++slice) {
        std::size_t slice_offset = slice * nx * ny;
        std::complex<grid_base_type> * offset_grid = grid + slice_offset;
	
	std::size_t half_x = nx / 2;
	std::size_t half_y = ny / 2;
	std::size_t odd_offset_x = (nx % 2 != 0) ? 1 : 0;
	//rotate all the rows right
	for (std::size_t iy =0; iy < ny; ++iy){
	    std::complex<grid_base_type> swap_mid = offset_grid[iy*nx + half_x];
	    for (std::size_t ix = 0; ix < half_x; ++ix){
		std::complex<grid_base_type> swap = offset_grid[iy*nx+ix]; //in case this dimension is even
		offset_grid[iy*nx+ix] = offset_grid[iy*nx + half_x + ix + odd_offset_x];
		offset_grid[iy*nx + half_x + ix] = swap;
	    }
	    if (nx % 2 != 0){
		offset_grid[iy*nx + nx - 1] = swap_mid;
	    }
	}
	std::size_t odd_offset_y = (ny % 2 != 0) ? 1 : 0;
	//rotate all the columns down
	for (std::size_t ix = 0; ix < nx; ++ix){
	    std::complex<grid_base_type> swap_mid = offset_grid[half_y*nx+ix];
	    for (std::size_t iy = 0; iy < half_y; ++iy){
		std::complex<grid_base_type> swap = offset_grid[iy*nx+ix]; //in case this dimension is even
		offset_grid[iy*nx+ix] = offset_grid[(iy + half_y + odd_offset_y)*nx + ix];
		offset_grid[(half_y + iy)*nx+ix] = swap;
	    }
	    if (ny % 2 != 0){
		offset_grid[(ny-1)*nx + ix] = swap_mid;
	    }
	}
    }
  }
  void ifftshift(std::complex<grid_base_type> * __restrict__ grid,
               std::size_t nx, std::size_t ny,
               std::size_t no_slices) {
    for (std::size_t slice = 0; slice < no_slices; ++slice) {
        std::size_t slice_offset = slice * nx * ny;
        std::complex<grid_base_type> * offset_grid = grid + slice_offset;
	std::size_t half_x = nx / 2;
	std::size_t half_y = ny / 2;
	std::size_t odd_offset_x = (nx % 2 != 0) ? 1 : 0;
	//rotate all the rows right
	for (std::size_t iy =0; iy < ny; ++iy){
	    std::complex<grid_base_type> swap_mid = offset_grid[iy*nx + half_x];
	    for (std::size_t ix = 0; ix < half_x; ++ix){
		std::size_t ix_reverse = half_x - 1 - ix;
		
		std::complex<grid_base_type> swap_x = offset_grid[iy*nx + half_x + ix_reverse + odd_offset_x];
		offset_grid[iy*nx + half_x + ix_reverse + odd_offset_x] = offset_grid[iy*nx + ix_reverse];
		offset_grid[iy*nx + ix_reverse + odd_offset_x] = swap_x;
	    }
	    offset_grid[iy*nx] = swap_mid; //doesn't matter for the even case
	}  
	std::size_t odd_offset_y = (ny % 2 != 0) ? 1 : 0;
	//rotate all the columns down
	for (std::size_t ix = 0; ix < nx; ++ix){
	    std::complex<grid_base_type> swap_mid = offset_grid[half_y*nx + ix];
	    for (std::size_t iy = 0; iy < half_y; ++iy){
		std::size_t iy_reverse = half_y - 1 - iy;
		std::complex<grid_base_type> swap_y = offset_grid[(half_y + iy_reverse + odd_offset_y)*nx + ix];
		offset_grid[(half_y + iy_reverse + odd_offset_y)*nx + ix] = offset_grid[iy_reverse*nx+ix];
		offset_grid[(iy_reverse+odd_offset_y)*nx + ix] = swap_y;
	    }
	    offset_grid[ix] = swap_mid; //doesn't matter for the even case
	}
    }
  }
}