#pragma once
#include <complex>
#include <cmath>
#include "base_types.h"

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
              std::size_t no_slices);

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
               std::size_t no_slices);
}