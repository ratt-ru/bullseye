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