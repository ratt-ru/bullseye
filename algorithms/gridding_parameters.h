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
// this based on your choice of grid_base_type
#define SHOULD_DO_32_BIT_FFT

struct gridding_parameters {
    //Mandatory data necessary for gridding:
    std::complex<visibility_base_type> * visibilities;
    imaging::uvw_coord<uvw_base_type> * uvw_coords;
    reference_wavelengths_base_type * reference_wavelengths;
    visibility_weights_base_type * visibility_weights;
    bool * flags;
    bool * flagged_rows;
    unsigned int * field_array;
    unsigned int * spw_index_array;
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
    std::complex<grid_base_type> * output_buffer;
    //Faceting information
    uvw_base_type phase_centre_ra;
    uvw_base_type phase_centre_dec;
    uvw_base_type * facet_centres;
    size_t num_facet_centres;
    //Fields required to specify jones facet_4_cor_corrections
    std::complex<visibility_base_type> * jones_terms;
    bool should_invert_jones_terms;
    unsigned int * antenna_1_ids;
    unsigned int * antenna_2_ids;
    std::size_t * timestamp_ids;
    size_t antenna_count;
    //Channel selection and averaging
    bool * enabled_channels;
    std::size_t * channel_grid_indicies;
    size_t cube_channel_dim_size;
    //Sampling function
    std::complex<grid_base_type> * sampling_function_buffer;
    std::size_t * sampling_function_channel_grid_indicies;
    size_t sampling_function_channel_count;
    //Precomputed Detapering coefficients
    convolution_base_type * detapering_buffer;
    //Finalization steps
    bool is_final_data_chunk;
    //baseline indexes needed for Romeins distribution strategy
    size_t * baseline_starting_indexes; //this has to be n(n-1)/2 + n + 1 long because we need to compute the length of the last baseline
};
