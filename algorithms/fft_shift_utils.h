#pragma once
#include <cmath>
#include "gridding_parameters.h"
namespace utils {
/**
 * FFT shift centres the DC component of an FFT to the centre position of an array
 * For a 2 dimensional FFT this relates to swapping quadrants 1 and 3, as well as
 * quadrants 2 and 4.
 * 
 * When doing this on an odd sized array the first ceil (n / 2) positions are 
 * shifted (ie. X != fftshift(fftshift(X)))
 */
void fftshift(std::complex<grid_base_type> * __restrict__ grid,
              std::size_t nx, std::size_t ny,
              std::size_t no_slices) {
    for (std::size_t slice = 0; slice < no_slices; ++slice) {
        std::size_t slice_offset = slice * nx * ny;
        std::size_t quad_size_x = ceil((nx) / 2.0);
        std::size_t quad_size_y = ceil((ny) / 2.0);
        std::size_t ubound = quad_size_x * quad_size_y;
        for (std::size_t i = 0; i < ubound; ++i) {
            std::size_t y_1 = i / quad_size_x;
            std::size_t x_1 = i - y_1 * quad_size_x;
            std::size_t y_3 = ny - (quad_size_y - y_1);
            std::size_t x_3 = nx - (quad_size_x - x_1);
            std::size_t x_2 = x_3;
            std::size_t y_2 = y_1;
            std::size_t x_4 = x_1;
            std::size_t y_4 = y_3;

            //swap quadrant 1 with quadrant 3 and swap quadrant 2 with quadrant 4:
            std::size_t quad_1_flat_index = y_1*nx + x_1 + slice_offset;
            std::size_t quad_3_flat_index = y_3*nx + x_3 + slice_offset;
            std::size_t quad_2_flat_index = y_2*nx + x_2 + slice_offset;
            std::size_t quad_4_flat_index = y_4*nx + x_4 + slice_offset;
            std::complex<grid_base_type> swap_1 = grid[quad_1_flat_index];
            std::complex<grid_base_type> swap_2 = grid[quad_2_flat_index];
            grid[quad_1_flat_index] = grid[quad_3_flat_index];
            grid[quad_2_flat_index] = grid[quad_4_flat_index];
            grid[quad_3_flat_index] = swap_1;
            grid[quad_4_flat_index] = swap_2;
        }
    }
}
/**
 * FFT shift centres the DC component of an FFT to the centre position of an array
 * For a 2 dimensional FFT this relates to swapping quadrants 1 and 3, as well as
 * quadrants 2 and 4.
 * 
 * When doing this on an odd sized array the first ceil (n / 2) positions are 
 * shifted (ie. X != ifftshift(ifftshift(X)))
 */
void ifftshift(std::complex<grid_base_type> * __restrict__ grid,
               std::size_t nx, std::size_t ny,
               std::size_t no_slices) {
    for (std::size_t slice = 0; slice < no_slices; ++slice) {
        std::size_t slice_offset = slice * nx * ny;
        std::size_t quad_size_x = (nx) / 2;
        std::size_t quad_size_y = (ny) / 2;
        std::size_t ubound = quad_size_x * quad_size_y;
        for (std::size_t i = 0; i < ubound; ++i) {
            std::size_t y_1 = i / quad_size_x;
            std::size_t x_1 = i - y_1 * quad_size_x;
            std::size_t y_3 = ny - (quad_size_y - y_1);
            std::size_t x_3 = nx - (quad_size_x - x_1);
	    
            std::size_t x_2 = x_3;
            std::size_t y_2 = y_1;
            std::size_t x_4 = x_1;
            std::size_t y_4 = y_3;

            //swap quadrant 1 with quadrant 3 and swap quadrant 2 with quadrant 4:
            std::size_t quad_1_flat_index = y_1*nx + x_1 + slice_offset;
            std::size_t quad_3_flat_index = y_3*nx + x_3 + slice_offset;
            std::size_t quad_2_flat_index = y_2*nx + x_2 + slice_offset;
            std::size_t quad_4_flat_index = y_4*nx + x_4 + slice_offset;
            std::complex<grid_base_type> swap_1 = grid[quad_1_flat_index];
            std::complex<grid_base_type> swap_2 = grid[quad_2_flat_index];
            grid[quad_1_flat_index] = grid[quad_3_flat_index];
            grid[quad_2_flat_index] = grid[quad_4_flat_index];
            grid[quad_3_flat_index] = swap_1;
            grid[quad_4_flat_index] = swap_2;
        }
    }
}
}