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
#include "uvw_coord.h"
#include "base_types.h"

struct gridding_parameters {
    //Mandatory data necessary for gridding:
    std::complex<visibility_base_type> * __restrict__  visibilities;
    imaging::uvw_coord<uvw_base_type> * __restrict__  uvw_coords;
    reference_wavelengths_base_type * __restrict__  reference_wavelengths;
    visibility_weights_base_type * __restrict__  visibility_weights;
    bool * __restrict__  flags;
    bool * __restrict__  flagged_rows;
    unsigned int * __restrict__  field_array;
    unsigned int * __restrict__  spw_index_array;
    unsigned int imaging_field; //mandatory: used to seperate different pointings in the MS 2.0 specification
    //Mandatory count fields necessary for gridding:
    size_t baseline_count;
    size_t row_count;
    size_t chunk_max_row_count; //this will be quite useful to preallocate all the space we can ever need on a gpu implementation
    size_t channel_count;
    size_t number_of_polarization_terms;
    size_t number_of_polarization_terms_being_gridded;
    size_t spw_count;
    size_t no_timestamps_read;
    //Mandatory image scaling fields necessary for scaling the IFFT correctly
    size_t nx;
    size_t ny;
    uvw_base_type cell_size_x;
    uvw_base_type cell_size_y;
    //Fields in use for specifying externally computed convolution function
    convolution_base_type * conv;
    size_t conv_support;
    size_t conv_oversample;
    //Correlation index specifier for gridding a single stokes/correlation term
    size_t polarization_index;
    size_t second_polarization_index;//only in use when gridding two correlation terms
    //Preallocated buffers
    std::complex<grid_base_type> * __restrict__ output_buffer;
    //Faceting information
    uvw_base_type phase_centre_ra;
    uvw_base_type phase_centre_dec;
    uvw_base_type * __restrict__ facet_centres;
    size_t num_facet_centres;
    //Fields required to specify jones facet_4_cor_corrections
    std::complex<visibility_base_type> * jones_terms;
    bool should_invert_jones_terms;
    unsigned int * __restrict__ antenna_1_ids;
    unsigned int * __restrict__ antenna_2_ids;
    std::size_t * __restrict__ timestamp_ids;
    size_t antenna_count;
    //Channel selection and averaging
    bool * __restrict__ enabled_channels;
    std::size_t * __restrict__ channel_grid_indicies;
    size_t cube_channel_dim_size;
    //Sampling function
    bool should_grid_sampling_function;
    std::complex<grid_base_type> * __restrict__ sampling_function_buffer;
    std::size_t * __restrict__ sampling_function_channel_grid_indicies;
    size_t sampling_function_channel_count;
    //Finalization steps
    bool is_final_data_chunk;
    //w-projection related terms
    std::size_t wplanes;
    uvw_base_type wmax_est;
    //baseline indexes needed for Romeins distribution strategy
    size_t * baseline_starting_indexes; //this has to be n(n-1)/2 + n + 1 long because we need to compute the length of the last baseline
    size_t * antenna_jones_starting_indexes; //this has to be n + 1 long because we need to be able to compute the number of jones terms at the last antenna
    size_t * jones_time_indicies_per_antenna; //this will be the same length as the repacked jones matrix array
    normalization_base_type * normalization_terms; //this has to be threads_bins x #facets x #channel_accumulation_grids x #polarization_being_gridded
};
