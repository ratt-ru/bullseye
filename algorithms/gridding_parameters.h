#pragma once
#include <complex>
#include "uvw_coord.h"

//just swap these for doubles if you're passing double precission numpy arrays through!
typedef float visibility_base_type;
typedef float uvw_base_type;
typedef float reference_wavelengths_base_type;
typedef float convolution_base_type;
typedef float visibility_weights_base_type;
typedef float grid_base_type;

struct gridding_parameters {
    //Mandatory data necessary for gridding:
    const std::complex<visibility_base_type> * visibilities;
    const imaging::uvw_coord<uvw_base_type> * uvw_coords;
    const reference_wavelengths_base_type * reference_wavelengths;
    const visibility_weights_base_type * visibility_weights;
    const bool * flags;
    const bool * flagged_rows;
    const unsigned int * field_array;
    const unsigned int * spw_index_array;
    unsigned int imaging_field; //mandatory: used to seperate different pointings in the MS 2.0 specification
    //Mandatory count fields necessary for gridding:
    size_t baseline_count;
    size_t row_count;
    size_t channel_count;
    size_t number_of_polarization_terms;
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
    std::complex<grid_base_type> * output_buffer;
    //Faceting information
    uvw_base_type phase_centre_ra;
    uvw_base_type phase_centre_dec;
    const uvw_base_type * facet_centres;
    size_t num_facet_centres;
    //Fields required to specify jones facet_4_cor_corrections
    std::complex<visibility_base_type> * jones_terms;
    bool should_invert_jones_terms;
    const unsigned int * antenna_1_ids;
    const unsigned int * antenna_2_ids;
    const std::size_t * timestamp_ids;
    size_t antenna_count;
    //Channel selection and averaging
    const bool * enabled_channels;
    const std::size_t * channel_grid_indicies;
    size_t cube_channel_dim_size;
};