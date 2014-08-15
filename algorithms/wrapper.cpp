#include <string>
#include <cstdio>
#include <casa/Quanta/Quantum.h>

#include "uvw_coord.h"
#include "baseline_transform_policies.h"
#include "phase_transform_policies.h"
#include "polarization_gridding_policies.h"
#include "convolution_policies.h"
#include "gridding.h"

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
  //Preallocated buffer to store the sampling function
  std::complex<grid_base_type> * sampling_function_buffer;
};

extern "C" {
    void grid_single_pol(gridding_parameters & params) {
        using namespace imaging;	
        printf("GRIDDING...");
        typedef baseline_transform_policy<uvw_base_type, transform_disable_facet_rotation> baseline_transform_policy_type;
        typedef phase_transform_policy<visibility_base_type,
                uvw_base_type,
                transform_disable_phase_rotation> phase_transform_policy_type;
        typedef polarization_gridding_policy<visibility_base_type, uvw_base_type,
                visibility_weights_base_type, convolution_base_type, grid_base_type,
                phase_transform_policy_type, gridding_single_pol> polarization_gridding_policy_type;
        typedef convolution_policy<convolution_base_type,uvw_base_type,grid_base_type,
                polarization_gridding_policy_type, convolution_precomputed_fir> convolution_policy_type;

        baseline_transform_policy_type uvw_transform; //standard: no uvw rotation
        phase_transform_policy_type phase_transform; //standard: no phase rotation

        polarization_gridding_policy_type polarization_policy(phase_transform,
                params.output_buffer,
                params.visibilities,
                params.visibility_weights,
                params.flags,
                params.number_of_polarization_terms,
                params.polarization_index,
                params.channel_count);
        convolution_policy_type convolution_policy(params.nx,params.ny,params.conv_support,params.conv_oversample,
                params.conv, polarization_policy);

        imaging::grid<visibility_base_type,uvw_base_type,
                reference_wavelengths_base_type,convolution_base_type,
                visibility_weights_base_type,grid_base_type,
                baseline_transform_policy_type,
                polarization_gridding_policy_type,
                convolution_policy_type>
                (polarization_policy,uvw_transform,convolution_policy,
                 params.uvw_coords,
                 params.flagged_rows,
                 params.nx,params.ny,
                 casa::Quantity(params.cell_size_x,"arcsec"),
                 casa::Quantity(params.cell_size_y,"arcsec"),
                 params.no_timestamps_read,params.baseline_count,params.channel_count,
                 params.row_count,params.reference_wavelengths,params.field_array,
                 params.imaging_field,params.spw_index_array);
        printf(" <DONE>\n");
    }

    void facet_single_pol(gridding_parameters & params) {
        using namespace imaging;
        size_t no_facet_pixels = params.nx*params.ny;
        for (size_t facet_index = 0; facet_index < params.num_facet_centres; ++facet_index) {
            uvw_base_type new_phase_ra = params.facet_centres[2*facet_index];
            uvw_base_type new_phase_dec = params.facet_centres[2*facet_index + 1];

            printf("FACETING (%f,%f,%f,%f) %lu / %lu...",params.phase_centre_ra,params.phase_centre_dec,new_phase_ra,new_phase_dec,facet_index+1, params.num_facet_centres);
            fflush(stdout);


            typedef imaging::baseline_transform_policy<uvw_base_type,
                    transform_facet_lefthanded_ra_dec> baseline_transform_policy_type;
            typedef imaging::phase_transform_policy<visibility_base_type,
                    uvw_base_type,
                    transform_enable_phase_rotation_lefthanded_ra_dec> phase_transform_policy_type;
            typedef imaging::polarization_gridding_policy<visibility_base_type, uvw_base_type,
                    visibility_weights_base_type, convolution_base_type, grid_base_type,
                    phase_transform_policy_type, gridding_single_pol> polarization_gridding_policy_type;
            typedef imaging::convolution_policy<convolution_base_type,uvw_base_type,grid_base_type,
                    polarization_gridding_policy_type,convolution_precomputed_fir> convolution_policy_type;
            baseline_transform_policy_type uvw_transform(0,0,casa::Quantity(params.phase_centre_ra,"arcsec"),casa::Quantity(params.phase_centre_dec,"arcsec"),
                    casa::Quantity(new_phase_ra,"arcsec"),casa::Quantity(new_phase_dec,"arcsec")); //lm faceting
            phase_transform_policy_type phase_transform(casa::Quantity(params.phase_centre_ra,"arcsec"),casa::Quantity(params.phase_centre_dec,"arcsec"),
                    casa::Quantity(new_phase_ra,"arcsec"),casa::Quantity(new_phase_dec,"arcsec")); //lm faceting

            polarization_gridding_policy_type polarization_policy(phase_transform,
                    params.output_buffer + no_facet_pixels*facet_index,
                    params.visibilities,
                    params.visibility_weights,
                    params.flags,
                    params.number_of_polarization_terms,
                    params.polarization_index,
                    params.channel_count);
            convolution_policy_type convolution_policy(params.nx,params.ny,
                    params.conv_support,params.conv_oversample,
                    params.conv, polarization_policy);
            imaging::grid<visibility_base_type,uvw_base_type,
                    reference_wavelengths_base_type,convolution_base_type,
                    visibility_weights_base_type,grid_base_type,
                    baseline_transform_policy_type,
                    polarization_gridding_policy_type,
                    convolution_policy_type>(polarization_policy,uvw_transform,convolution_policy,
                                             params.uvw_coords,
                                             params.flagged_rows,
                                             params.nx,params.ny,
                                             casa::Quantity(params.cell_size_x,"arcsec"),casa::Quantity(params.cell_size_y,"arcsec"),
                                             params.no_timestamps_read,params.baseline_count,params.channel_count,
                                             params.row_count,params.reference_wavelengths,params.field_array,
                                             params.imaging_field,params.spw_index_array);
            printf(" <DONE>\n");
        }
    }
    void grid_4_cor(gridding_parameters & params) {
        using namespace imaging;
        assert(params.number_of_polarization_terms == 4); //Only supports 4 correlation visibilties in this mode
        printf("GRIDDING...");
        typedef baseline_transform_policy<uvw_base_type, transform_disable_facet_rotation> baseline_transform_policy_type;
        typedef phase_transform_policy<visibility_base_type,
                uvw_base_type,
                transform_disable_phase_rotation> phase_transform_policy_type;
        typedef polarization_gridding_policy<visibility_base_type, uvw_base_type,
                visibility_weights_base_type, convolution_base_type, grid_base_type,
                phase_transform_policy_type, gridding_4_pol> polarization_gridding_policy_type;
        typedef convolution_policy<convolution_base_type,uvw_base_type,grid_base_type,
                polarization_gridding_policy_type, convolution_precomputed_fir> convolution_policy_type;

        baseline_transform_policy_type uvw_transform; //standard: no uvw rotation
        phase_transform_policy_type phase_transform; //standard: no phase rotation

        polarization_gridding_policy_type polarization_policy(phase_transform,
                params.output_buffer,
                params.visibilities,
                params.visibility_weights,
                params.flags,params.nx*params.ny,
                params.channel_count);
        convolution_policy_type convolution_policy(params.nx,params.ny,params.conv_support,params.conv_oversample,
                params.conv, polarization_policy);

        imaging::grid<visibility_base_type,uvw_base_type,
                reference_wavelengths_base_type,convolution_base_type,
                visibility_weights_base_type,grid_base_type,
                baseline_transform_policy_type,
                polarization_gridding_policy_type,
                convolution_policy_type>
                (polarization_policy,uvw_transform,convolution_policy,
                 params.uvw_coords,
                 params.flagged_rows,
                 params.nx,params.ny,
                 casa::Quantity(params.cell_size_x,"arcsec"),
                 casa::Quantity(params.cell_size_y,"arcsec"),
                 params.no_timestamps_read,params.baseline_count,params.channel_count,
                 params.row_count,params.reference_wavelengths,params.field_array,
                 params.imaging_field,params.spw_index_array);
        printf(" <DONE>\n");
    }
    void facet_4_cor(gridding_parameters & params) {
        using namespace imaging;
        assert(params.number_of_polarization_terms == 4); //Only supports 4 correlation visibilties in this mode

        size_t no_facet_pixels = params.nx*params.ny;
        for (size_t facet_index = 0; facet_index < params.num_facet_centres; ++facet_index) {
            uvw_base_type new_phase_ra = params.facet_centres[2*facet_index];
            uvw_base_type new_phase_dec = params.facet_centres[2*facet_index + 1];

            printf("FACETING (%f,%f,%f,%f) %lu / %lu...",params.phase_centre_ra,params.phase_centre_dec,new_phase_ra,new_phase_dec,facet_index+1, params.num_facet_centres);
            fflush(stdout);

            typedef imaging::baseline_transform_policy<uvw_base_type,
                    transform_facet_lefthanded_ra_dec> baseline_transform_policy_type;
            typedef imaging::phase_transform_policy<visibility_base_type,
                    uvw_base_type,
                    transform_enable_phase_rotation_lefthanded_ra_dec> phase_transform_policy_type;
            typedef imaging::polarization_gridding_policy<visibility_base_type, uvw_base_type,
                    visibility_weights_base_type, convolution_base_type, grid_base_type,
                    phase_transform_policy_type, gridding_4_pol> polarization_gridding_policy_type;
            typedef imaging::convolution_policy<convolution_base_type,uvw_base_type,grid_base_type,
                    polarization_gridding_policy_type,convolution_precomputed_fir> convolution_policy_type;

            baseline_transform_policy_type uvw_transform(0,0,casa::Quantity(params.phase_centre_ra,"arcsec"),casa::Quantity(params.phase_centre_dec,"arcsec"),
                    casa::Quantity(new_phase_ra,"arcsec"),casa::Quantity(new_phase_dec,"arcsec")); //lm faceting
            //baseline_transform_policy_type uvw_transform; //uv faceting
            phase_transform_policy_type phase_transform(casa::Quantity(params.phase_centre_ra,"arcsec"),casa::Quantity(params.phase_centre_dec,"arcsec"),
                    casa::Quantity(new_phase_ra,"arcsec"),casa::Quantity(new_phase_dec,"arcsec")); //lm faceting

            polarization_gridding_policy_type polarization_policy(phase_transform,
                    params.output_buffer + no_facet_pixels*facet_index*params.number_of_polarization_terms,
                    params.visibilities,
                    params.visibility_weights,
                    params.flags,params.nx*params.ny,
                    params.channel_count);
            convolution_policy_type convolution_policy(params.nx,params.ny,
                    params.conv_support,params.conv_oversample,
                    params.conv, polarization_policy);
            imaging::grid<visibility_base_type,uvw_base_type,
                    reference_wavelengths_base_type,convolution_base_type,
                    visibility_weights_base_type,grid_base_type,
                    baseline_transform_policy_type,
                    polarization_gridding_policy_type,
                    convolution_policy_type>(polarization_policy,uvw_transform,convolution_policy,
                                             params.uvw_coords,
                                             params.flagged_rows,
                                             params.nx,params.ny,
                                             casa::Quantity(params.cell_size_x,"arcsec"),casa::Quantity(params.cell_size_y,"arcsec"),
                                             params.no_timestamps_read,params.baseline_count,params.channel_count,
                                             params.row_count,params.reference_wavelengths,params.field_array,
                                             params.imaging_field,params.spw_index_array);
            printf(" <DONE>\n");
        }
    }
    void facet_4_cor_corrections(gridding_parameters & params) {
        using namespace imaging;
        assert(params.number_of_polarization_terms == 4); //Only supports 4 correlation visibilties in this mode

        size_t no_facet_pixels = params.nx*params.ny;
	if (params.should_invert_jones_terms)
	      printf("INVERTING %zu JONES MATRICIES...\n", 
		     params.no_timestamps_read*params.antenna_count*params.num_facet_centres*params.spw_count*params.channel_count);
	      invert_all<visibility_base_type>((jones_2x2<visibility_base_type>*)(params.jones_terms),
					       params.no_timestamps_read*params.antenna_count*params.num_facet_centres*params.spw_count*params.channel_count);
        for (size_t facet_index = 0; facet_index < params.num_facet_centres; ++facet_index) {
            uvw_base_type new_phase_ra = params.facet_centres[2*facet_index];
            uvw_base_type new_phase_dec = params.facet_centres[2*facet_index + 1];

            printf("FACETING (%f,%f,%f,%f) %lu / %lu...",params.phase_centre_ra,params.phase_centre_dec,new_phase_ra,new_phase_dec,facet_index+1, params.num_facet_centres);
            fflush(stdout);

            typedef imaging::baseline_transform_policy<uvw_base_type,
                    transform_facet_lefthanded_ra_dec> baseline_transform_policy_type;
            typedef imaging::phase_transform_policy<visibility_base_type,
                    uvw_base_type,
                    transform_enable_phase_rotation_lefthanded_ra_dec> phase_transform_policy_type;
            typedef imaging::polarization_gridding_policy<visibility_base_type, uvw_base_type,
                    visibility_weights_base_type, convolution_base_type, grid_base_type,
                    phase_transform_policy_type, gridding_4_pol_enable_facet_based_jones_corrections> polarization_gridding_policy_type;
            typedef imaging::convolution_policy<convolution_base_type,uvw_base_type,grid_base_type,
                    polarization_gridding_policy_type,convolution_precomputed_fir> convolution_policy_type;

            baseline_transform_policy_type uvw_transform(0,0,casa::Quantity(params.phase_centre_ra,"arcsec"),casa::Quantity(params.phase_centre_dec,"arcsec"),
                    casa::Quantity(new_phase_ra,"arcsec"),casa::Quantity(new_phase_dec,"arcsec")); //lm faceting
            //baseline_transform_policy_type uvw_transform; //uv faceting
            phase_transform_policy_type phase_transform(casa::Quantity(params.phase_centre_ra,"arcsec"),casa::Quantity(params.phase_centre_dec,"arcsec"),
                    casa::Quantity(new_phase_ra,"arcsec"),casa::Quantity(new_phase_dec,"arcsec")); //lm faceting
            polarization_gridding_policy_type polarization_policy(phase_transform,
                    params.output_buffer + no_facet_pixels*facet_index*params.number_of_polarization_terms,
                    params.visibilities,
                    params.visibility_weights,
                    params.flags,params.nx*params.ny,
                    params.channel_count,
		    (jones_2x2<visibility_base_type>*)(params.jones_terms),
		    params.antenna_1_ids,
		    params.antenna_2_ids,
		    params.timestamp_ids,
		    params.antenna_count,
		    facet_index,
		    params.num_facet_centres,
		    params.no_timestamps_read,
		    params.spw_count
	    );
            convolution_policy_type convolution_policy(params.nx,params.ny,
                    params.conv_support,params.conv_oversample,
                    params.conv, polarization_policy);
            imaging::grid<visibility_base_type,uvw_base_type,
                    reference_wavelengths_base_type,convolution_base_type,
                    visibility_weights_base_type,grid_base_type,
                    baseline_transform_policy_type,
                    polarization_gridding_policy_type,
                    convolution_policy_type>(polarization_policy,uvw_transform,convolution_policy,
                                             params.uvw_coords,
                                             params.flagged_rows,
                                             params.nx,params.ny,
                                             casa::Quantity(params.cell_size_x,"arcsec"),casa::Quantity(params.cell_size_y,"arcsec"),
                                             params.no_timestamps_read,params.baseline_count,params.channel_count,
                                             params.row_count,params.reference_wavelengths,params.field_array,
                                             params.imaging_field,params.spw_index_array);
            printf(" <DONE>\n");
        }
    }
    
    void grid_single_pol_with_sampling_func(gridding_parameters & params) {
        using namespace imaging;

        printf("GRIDDING...");
        typedef baseline_transform_policy<uvw_base_type, transform_disable_facet_rotation> baseline_transform_policy_type;
        typedef phase_transform_policy<visibility_base_type,
                uvw_base_type,
                transform_disable_phase_rotation> phase_transform_policy_type;
        typedef polarization_gridding_policy<visibility_base_type, uvw_base_type,
                visibility_weights_base_type, convolution_base_type, grid_base_type,
                phase_transform_policy_type, gridding_single_pol> polarization_gridding_policy_type;
        typedef convolution_policy<convolution_base_type,uvw_base_type,grid_base_type,
                polarization_gridding_policy_type, convolution_with_sampling_function_gridding_using_precomputed_fir > convolution_policy_type;

        baseline_transform_policy_type uvw_transform; //standard: no uvw rotation
        phase_transform_policy_type phase_transform; //standard: no phase rotation

        polarization_gridding_policy_type polarization_policy(phase_transform,
                params.output_buffer,
                params.visibilities,
                params.visibility_weights,
                params.flags,
                params.number_of_polarization_terms,
                params.polarization_index,
                params.channel_count);
        convolution_policy_type convolution_policy(params.nx,params.ny,params.conv_support,params.conv_oversample,
                params.conv, polarization_policy,params.sampling_function_buffer);

        imaging::grid<visibility_base_type,uvw_base_type,
                reference_wavelengths_base_type,convolution_base_type,
                visibility_weights_base_type,grid_base_type,
                baseline_transform_policy_type,
                polarization_gridding_policy_type,
                convolution_policy_type>
                (polarization_policy,uvw_transform,convolution_policy,
                 params.uvw_coords,
                 params.flagged_rows,
                 params.nx,params.ny,
                 casa::Quantity(params.cell_size_x,"arcsec"),
                 casa::Quantity(params.cell_size_y,"arcsec"),
                 params.no_timestamps_read,params.baseline_count,params.channel_count,
                 params.row_count,params.reference_wavelengths,params.field_array,
                 params.imaging_field,params.spw_index_array);
        printf(" <DONE>\n");
    }

    void facet_single_pol_with_sampling_func(gridding_parameters & params) {
        using namespace imaging;
        size_t no_facet_pixels = params.nx*params.ny;
        for (size_t facet_index = 0; facet_index < params.num_facet_centres; ++facet_index) {
            uvw_base_type new_phase_ra = params.facet_centres[2*facet_index];
            uvw_base_type new_phase_dec = params.facet_centres[2*facet_index + 1];

            printf("FACETING (%f,%f,%f,%f) %lu / %lu...",params.phase_centre_ra,params.phase_centre_dec,new_phase_ra,new_phase_dec,facet_index+1, params.num_facet_centres);
            fflush(stdout);
	    
            typedef imaging::baseline_transform_policy<uvw_base_type,
                    transform_facet_lefthanded_ra_dec> baseline_transform_policy_type;
            typedef imaging::phase_transform_policy<visibility_base_type,
                    uvw_base_type,
                    transform_enable_phase_rotation_lefthanded_ra_dec> phase_transform_policy_type;
            typedef imaging::polarization_gridding_policy<visibility_base_type, uvw_base_type,
                    visibility_weights_base_type, convolution_base_type, grid_base_type,
                    phase_transform_policy_type, gridding_single_pol> polarization_gridding_policy_type;
            typedef imaging::convolution_policy<convolution_base_type,uvw_base_type,grid_base_type,
                    polarization_gridding_policy_type,convolution_with_sampling_function_gridding_using_precomputed_fir> convolution_policy_type;
            baseline_transform_policy_type uvw_transform(0,0,casa::Quantity(params.phase_centre_ra,"arcsec"),casa::Quantity(params.phase_centre_dec,"arcsec"),
                    casa::Quantity(new_phase_ra,"arcsec"),casa::Quantity(new_phase_dec,"arcsec")); //lm faceting
            phase_transform_policy_type phase_transform(casa::Quantity(params.phase_centre_ra,"arcsec"),casa::Quantity(params.phase_centre_dec,"arcsec"),
                    casa::Quantity(new_phase_ra,"arcsec"),casa::Quantity(new_phase_dec,"arcsec")); //lm faceting

            polarization_gridding_policy_type polarization_policy(phase_transform,
                    params.output_buffer + no_facet_pixels*facet_index,
                    params.visibilities,
                    params.visibility_weights,
                    params.flags,
                    params.number_of_polarization_terms,
                    params.polarization_index,
                    params.channel_count);
            convolution_policy_type convolution_policy(params.nx,params.ny,
                    params.conv_support,params.conv_oversample,
                    params.conv, polarization_policy,params.sampling_function_buffer + no_facet_pixels*facet_index);
            imaging::grid<visibility_base_type,uvw_base_type,
                    reference_wavelengths_base_type,convolution_base_type,
                    visibility_weights_base_type,grid_base_type,
                    baseline_transform_policy_type,
                    polarization_gridding_policy_type,
                    convolution_policy_type>(polarization_policy,uvw_transform,convolution_policy,
                                             params.uvw_coords,
                                             params.flagged_rows,
                                             params.nx,params.ny,
                                             casa::Quantity(params.cell_size_x,"arcsec"),casa::Quantity(params.cell_size_y,"arcsec"),
                                             params.no_timestamps_read,params.baseline_count,params.channel_count,
                                             params.row_count,params.reference_wavelengths,params.field_array,
                                             params.imaging_field,params.spw_index_array);
            printf(" <DONE>\n");
        }
    }
    void grid_4_cor_with_sampling_func(gridding_parameters & params) {
        using namespace imaging;
        assert(params.number_of_polarization_terms == 4); //Only supports 4 correlation visibilties in this mode
        printf("GRIDDING...");
        typedef baseline_transform_policy<uvw_base_type, transform_disable_facet_rotation> baseline_transform_policy_type;
        typedef phase_transform_policy<visibility_base_type,
                uvw_base_type,
                transform_disable_phase_rotation> phase_transform_policy_type;
        typedef polarization_gridding_policy<visibility_base_type, uvw_base_type,
                visibility_weights_base_type, convolution_base_type, grid_base_type,
                phase_transform_policy_type, gridding_4_pol> polarization_gridding_policy_type;
        typedef convolution_policy<convolution_base_type,uvw_base_type,grid_base_type,
                polarization_gridding_policy_type, convolution_with_sampling_function_gridding_using_precomputed_fir> convolution_policy_type;

        baseline_transform_policy_type uvw_transform; //standard: no uvw rotation
        phase_transform_policy_type phase_transform; //standard: no phase rotation

        polarization_gridding_policy_type polarization_policy(phase_transform,
                params.output_buffer,
                params.visibilities,
                params.visibility_weights,
                params.flags,params.nx*params.ny,
                params.channel_count);
        convolution_policy_type convolution_policy(params.nx,params.ny,params.conv_support,params.conv_oversample,
                params.conv, polarization_policy,params.sampling_function_buffer);

        imaging::grid<visibility_base_type,uvw_base_type,
                reference_wavelengths_base_type,convolution_base_type,
                visibility_weights_base_type,grid_base_type,
                baseline_transform_policy_type,
                polarization_gridding_policy_type,
                convolution_policy_type>
                (polarization_policy,uvw_transform,convolution_policy,
                 params.uvw_coords,
                 params.flagged_rows,
                 params.nx,params.ny,
                 casa::Quantity(params.cell_size_x,"arcsec"),
                 casa::Quantity(params.cell_size_y,"arcsec"),
                 params.no_timestamps_read,params.baseline_count,params.channel_count,
                 params.row_count,params.reference_wavelengths,params.field_array,
                 params.imaging_field,params.spw_index_array);
        printf(" <DONE>\n");
    }
    void facet_4_cor_with_sampling_func(gridding_parameters & params) {
        using namespace imaging;
        assert(params.number_of_polarization_terms == 4); //Only supports 4 correlation visibilties in this mode

        size_t no_facet_pixels = params.nx*params.ny;
        for (size_t facet_index = 0; facet_index < params.num_facet_centres; ++facet_index) {
            uvw_base_type new_phase_ra = params.facet_centres[2*facet_index];
            uvw_base_type new_phase_dec = params.facet_centres[2*facet_index + 1];

            printf("FACETING (%f,%f,%f,%f) %lu / %lu...",params.phase_centre_ra,params.phase_centre_dec,new_phase_ra,new_phase_dec,facet_index+1, params.num_facet_centres);
            fflush(stdout);

            typedef imaging::baseline_transform_policy<uvw_base_type,
                    transform_facet_lefthanded_ra_dec> baseline_transform_policy_type;
            typedef imaging::phase_transform_policy<visibility_base_type,
                    uvw_base_type,
                    transform_enable_phase_rotation_lefthanded_ra_dec> phase_transform_policy_type;
            typedef imaging::polarization_gridding_policy<visibility_base_type, uvw_base_type,
                    visibility_weights_base_type, convolution_base_type, grid_base_type,
                    phase_transform_policy_type, gridding_4_pol> polarization_gridding_policy_type;
            typedef imaging::convolution_policy<convolution_base_type,uvw_base_type,grid_base_type,
                    polarization_gridding_policy_type,convolution_with_sampling_function_gridding_using_precomputed_fir> convolution_policy_type;

            baseline_transform_policy_type uvw_transform(0,0,casa::Quantity(params.phase_centre_ra,"arcsec"),casa::Quantity(params.phase_centre_dec,"arcsec"),
                    casa::Quantity(new_phase_ra,"arcsec"),casa::Quantity(new_phase_dec,"arcsec")); //lm faceting
            //baseline_transform_policy_type uvw_transform; //uv faceting
            phase_transform_policy_type phase_transform(casa::Quantity(params.phase_centre_ra,"arcsec"),casa::Quantity(params.phase_centre_dec,"arcsec"),
                    casa::Quantity(new_phase_ra,"arcsec"),casa::Quantity(new_phase_dec,"arcsec")); //lm faceting

            polarization_gridding_policy_type polarization_policy(phase_transform,
                    params.output_buffer + no_facet_pixels*facet_index*params.number_of_polarization_terms,
                    params.visibilities,
                    params.visibility_weights,
                    params.flags,params.nx*params.ny,
                    params.channel_count);
            convolution_policy_type convolution_policy(params.nx,params.ny,
                    params.conv_support,params.conv_oversample,
                    params.conv, polarization_policy, params.sampling_function_buffer + no_facet_pixels*facet_index); //only one sampling function over all 4 polarizations
            imaging::grid<visibility_base_type,uvw_base_type,
                    reference_wavelengths_base_type,convolution_base_type,
                    visibility_weights_base_type,grid_base_type,
                    baseline_transform_policy_type,
                    polarization_gridding_policy_type,
                    convolution_policy_type>(polarization_policy,uvw_transform,convolution_policy,
                                             params.uvw_coords,
                                             params.flagged_rows,
                                             params.nx,params.ny,
                                             casa::Quantity(params.cell_size_x,"arcsec"),casa::Quantity(params.cell_size_y,"arcsec"),
                                             params.no_timestamps_read,params.baseline_count,params.channel_count,
                                             params.row_count,params.reference_wavelengths,params.field_array,
                                             params.imaging_field,params.spw_index_array);
            printf(" <DONE>\n");
        }
    }
    void facet_4_cor_corrections_with_sampling_func(gridding_parameters & params) {
        using namespace imaging;
        assert(params.number_of_polarization_terms == 4); //Only supports 4 correlation visibilties in this mode

        size_t no_facet_pixels = params.nx*params.ny;
	if (params.should_invert_jones_terms)
	      printf("INVERTING %zu JONES MATRICIES...\n", 
		     params.no_timestamps_read*params.antenna_count*params.num_facet_centres*params.spw_count*params.channel_count);
	      invert_all<visibility_base_type>((jones_2x2<visibility_base_type>*)(params.jones_terms),
					       params.no_timestamps_read*params.antenna_count*params.num_facet_centres*params.spw_count*params.channel_count);
        for (size_t facet_index = 0; facet_index < params.num_facet_centres; ++facet_index) {
            uvw_base_type new_phase_ra = params.facet_centres[2*facet_index];
            uvw_base_type new_phase_dec = params.facet_centres[2*facet_index + 1];

            printf("FACETING (%f,%f,%f,%f) %lu / %lu...",params.phase_centre_ra,params.phase_centre_dec,new_phase_ra,new_phase_dec,facet_index+1, params.num_facet_centres);
            fflush(stdout);

            typedef imaging::baseline_transform_policy<uvw_base_type,
                    transform_facet_lefthanded_ra_dec> baseline_transform_policy_type;
            typedef imaging::phase_transform_policy<visibility_base_type,
                    uvw_base_type,
                    transform_enable_phase_rotation_lefthanded_ra_dec> phase_transform_policy_type;
            typedef imaging::polarization_gridding_policy<visibility_base_type, uvw_base_type,
                    visibility_weights_base_type, convolution_base_type, grid_base_type,
                    phase_transform_policy_type, gridding_4_pol_enable_facet_based_jones_corrections> polarization_gridding_policy_type;
            typedef imaging::convolution_policy<convolution_base_type,uvw_base_type,grid_base_type,
                    polarization_gridding_policy_type,convolution_with_sampling_function_gridding_using_precomputed_fir> convolution_policy_type;

            baseline_transform_policy_type uvw_transform(0,0,casa::Quantity(params.phase_centre_ra,"arcsec"),casa::Quantity(params.phase_centre_dec,"arcsec"),
                    casa::Quantity(new_phase_ra,"arcsec"),casa::Quantity(new_phase_dec,"arcsec")); //lm faceting
            //baseline_transform_policy_type uvw_transform; //uv faceting
            phase_transform_policy_type phase_transform(casa::Quantity(params.phase_centre_ra,"arcsec"),casa::Quantity(params.phase_centre_dec,"arcsec"),
                    casa::Quantity(new_phase_ra,"arcsec"),casa::Quantity(new_phase_dec,"arcsec")); //lm faceting
            polarization_gridding_policy_type polarization_policy(phase_transform,
                    params.output_buffer + no_facet_pixels*facet_index*params.number_of_polarization_terms,
                    params.visibilities,
                    params.visibility_weights,
                    params.flags,params.nx*params.ny,
                    params.channel_count,
		    (jones_2x2<visibility_base_type>*)(params.jones_terms),
		    params.antenna_1_ids,
		    params.antenna_2_ids,
		    params.timestamp_ids,
		    params.antenna_count,
		    facet_index,
		    params.num_facet_centres,
		    params.no_timestamps_read,
		    params.spw_count
	    );
            convolution_policy_type convolution_policy(params.nx,params.ny,
                    params.conv_support,params.conv_oversample,
                    params.conv, polarization_policy,params.sampling_function_buffer + no_facet_pixels*facet_index); //only one sampling function over all 4 polarizations
            imaging::grid<visibility_base_type,uvw_base_type,
                    reference_wavelengths_base_type,convolution_base_type,
                    visibility_weights_base_type,grid_base_type,
                    baseline_transform_policy_type,
                    polarization_gridding_policy_type,
                    convolution_policy_type>(polarization_policy,uvw_transform,convolution_policy,
                                             params.uvw_coords,
                                             params.flagged_rows,
                                             params.nx,params.ny,
                                             casa::Quantity(params.cell_size_x,"arcsec"),casa::Quantity(params.cell_size_y,"arcsec"),
                                             params.no_timestamps_read,params.baseline_count,params.channel_count,
                                             params.row_count,params.reference_wavelengths,params.field_array,
                                             params.imaging_field,params.spw_index_array);
            printf(" <DONE>\n");
        }
    }
}
