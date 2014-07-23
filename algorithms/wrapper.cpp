#include <string>
#include <cstdio>

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
extern "C" {
  void grid_single_pol(const std::complex<visibility_base_type> * visibilities, const imaging::uvw_coord<uvw_base_type> * uvw_coords,
		       size_t timestamp_count, size_t baseline_count, size_t channel_count, size_t number_of_polarization_terms,
		       const reference_wavelengths_base_type * reference_wavelengths, const bool * flags, const bool * flagged_rows,
		       visibility_weights_base_type *  visibility_weights, size_t facet_nx, size_t facet_ny, uvw_base_type facet_cell_size_x,
		       uvw_base_type facet_cell_size_y, uvw_base_type phase_centre_ra, uvw_base_type phase_centre_dec, const uvw_base_type * facet_centres, 
		       std::size_t num_facet_centres, convolution_base_type * conv, size_t conv_support,size_t conv_oversample,size_t polarization_index,
		       std::complex<grid_base_type> * output_buffer, size_t row_count, const unsigned int * field_array,
		       unsigned int imaging_field){
    using namespace imaging;
   
    if (facet_centres == nullptr){ //no faceting
		printf("GRIDDING...");
		typedef baseline_transform_policy<uvw_base_type, transform_disable_facet_rotation> baseline_transform_policy_type;
		typedef phase_transform_policy<visibility_base_type, 
					       uvw_base_type, 
					       transform_disable_phase_rotation> phase_transform_policy_type;
		typedef polarization_gridding_policy<visibility_base_type, uvw_base_type, 
						     visibility_weights_base_type, convolution_base_type, grid_base_type,
						     phase_transform_policy_type, gridding_single_pol> polarization_gridding_policy_type;
		typedef convolution_policy<convolution_base_type,uvw_base_type,
					   polarization_gridding_policy_type, convolution_precomputed_fir> convolution_policy_type;
			
			baseline_transform_policy_type uvw_transform; //standard: no uvw rotation
			phase_transform_policy_type phase_transform; //standard: no phase rotation

			polarization_gridding_policy_type polarization_policy(phase_transform,
									      output_buffer,
									      visibilities,
									      visibility_weights,
									      flags,
									      number_of_polarization_terms,polarization_index);
			convolution_policy_type convolution_policy(facet_nx,facet_ny,conv_support,conv_oversample,
								   conv, polarization_policy);
			
			imaging::grid<visibility_base_type,uvw_base_type,
				      reference_wavelengths_base_type,convolution_base_type,
				      visibility_weights_base_type,grid_base_type,
				      baseline_transform_policy_type,
				      polarization_gridding_policy_type,
				      convolution_policy_type>
							     (polarization_policy,uvw_transform,convolution_policy,
                                        	     	      uvw_coords,
							      flagged_rows,
	                                             	      facet_nx,facet_ny,
        	                                     	      casa::Quantity(facet_cell_size_x,"arcsec"),
							      casa::Quantity(facet_cell_size_y,"arcsec"),
                	                             	      timestamp_count,baseline_count,channel_count,
							      row_count,reference_wavelengths,field_array,
							      imaging_field);
		printf(" <DONE>\n");
    } else { //enable faceting
	      size_t no_facet_pixels = facet_nx*facet_ny;
	      for (size_t facet_index = 0; facet_index < num_facet_centres; ++facet_index){
			uvw_base_type new_phase_ra = facet_centres[2*facet_index];
			uvw_base_type new_phase_dec = facet_centres[2*facet_index + 1];
			
			printf("FACETING (%f,%f,%f,%f) %lu / %lu...",phase_centre_ra,phase_centre_dec,new_phase_ra,new_phase_dec,facet_index+1, num_facet_centres);
			fflush(stdout);
			
			
			typedef imaging::baseline_transform_policy<uvw_base_type, 
								   transform_facet_righthanded_ra_dec> baseline_transform_policy_type;
			typedef imaging::phase_transform_policy<visibility_base_type, 
								uvw_base_type, 
								transform_enable_phase_rotation_righthanded_ra_dec> phase_transform_policy_type;
			typedef imaging::polarization_gridding_policy<visibility_base_type, uvw_base_type, 
								      visibility_weights_base_type, convolution_base_type, grid_base_type,
								      phase_transform_policy_type, gridding_single_pol> polarization_gridding_policy_type;
			typedef imaging::convolution_policy<convolution_base_type,uvw_base_type,
							    polarization_gridding_policy_type,convolution_precomputed_fir> convolution_policy_type;
			baseline_transform_policy_type uvw_transform(0,0,casa::Quantity(phase_centre_ra,"arcsec"),casa::Quantity(phase_centre_dec,"arcsec"),
								     casa::Quantity(new_phase_ra,"arcsec"),casa::Quantity(new_phase_dec,"arcsec")); //lm faceting
			//baseline_transform_policy_type uvw_transform; //uv faceting
			phase_transform_policy_type phase_transform(casa::Quantity(phase_centre_ra,"arcsec"),casa::Quantity(phase_centre_dec,"arcsec"),
								    casa::Quantity(new_phase_ra,"arcsec"),casa::Quantity(new_phase_dec,"arcsec")); //lm faceting
			
			polarization_gridding_policy_type polarization_policy(phase_transform,
									      output_buffer + no_facet_pixels*facet_index,
									      visibilities,
									      visibility_weights,
									      flags,
									      number_of_polarization_terms,polarization_index);
			convolution_policy_type convolution_policy(facet_nx,facet_ny,
								   conv_support,conv_oversample,
								   conv, polarization_policy);
			imaging::grid<visibility_base_type,uvw_base_type,
				      reference_wavelengths_base_type,convolution_base_type,
				      visibility_weights_base_type,grid_base_type,
				      baseline_transform_policy_type,
				      polarization_gridding_policy_type,
				      convolution_policy_type>(polarization_policy,uvw_transform,convolution_policy,
									  uvw_coords,
									  flagged_rows,
									  facet_nx,facet_ny,
									  casa::Quantity(facet_cell_size_x,"arcsec"),casa::Quantity(facet_cell_size_y,"arcsec"),
									  timestamp_count,baseline_count,channel_count,
									  row_count,reference_wavelengths,field_array,
									  imaging_field);
			printf(" <DONE>\n");	
		}
    }
  }
  void grid_4_cor(const std::complex<visibility_base_type> * visibilities, const imaging::uvw_coord<uvw_base_type> * uvw_coords,
		  size_t timestamp_count, size_t baseline_count, size_t channel_count, size_t number_of_polarization_terms,
		  const reference_wavelengths_base_type * reference_wavelengths, const bool * flags, const bool * flagged_rows,
		  visibility_weights_base_type *  visibility_weights, size_t facet_nx, size_t facet_ny, uvw_base_type facet_cell_size_x,
		  uvw_base_type facet_cell_size_y, uvw_base_type phase_centre_ra, uvw_base_type phase_centre_dec, const uvw_base_type * facet_centres, 
		  std::size_t num_facet_centres, convolution_base_type * conv, size_t conv_support,size_t conv_oversample,
		  std::complex<grid_base_type> * output_buffer, size_t row_count, const unsigned int * field_array,
		  unsigned int imaging_field){
    using namespace imaging;
    assert(number_of_polarization_terms == 4); //Only supports 4 correlation visibilties in this mode
    if (facet_centres == nullptr){ //no faceting
		printf("GRIDDING...");
		typedef baseline_transform_policy<uvw_base_type, transform_disable_facet_rotation> baseline_transform_policy_type;
		typedef phase_transform_policy<visibility_base_type, 
					       uvw_base_type, 
					       transform_disable_phase_rotation> phase_transform_policy_type;
		typedef polarization_gridding_policy<visibility_base_type, uvw_base_type, 
						     visibility_weights_base_type, convolution_base_type, grid_base_type,
						     phase_transform_policy_type, gridding_4_pol> polarization_gridding_policy_type;
		typedef convolution_policy<convolution_base_type,uvw_base_type,
					   polarization_gridding_policy_type, convolution_precomputed_fir> convolution_policy_type;
			
			baseline_transform_policy_type uvw_transform; //standard: no uvw rotation
			phase_transform_policy_type phase_transform; //standard: no phase rotation

			polarization_gridding_policy_type polarization_policy(phase_transform,
									      output_buffer,
									      visibilities,
									      visibility_weights,
									      flags);
			convolution_policy_type convolution_policy(facet_nx,facet_ny,conv_support,conv_oversample,
								   conv, polarization_policy);
			
			imaging::grid<visibility_base_type,uvw_base_type,
				      reference_wavelengths_base_type,convolution_base_type,
				      visibility_weights_base_type,grid_base_type,
				      baseline_transform_policy_type,
				      polarization_gridding_policy_type,
				      convolution_policy_type>
							     (polarization_policy,uvw_transform,convolution_policy,
                                        	     	      uvw_coords,
							      flagged_rows,
	                                             	      facet_nx,facet_ny,
        	                                     	      casa::Quantity(facet_cell_size_x,"arcsec"),
							      casa::Quantity(facet_cell_size_y,"arcsec"),
                	                             	      timestamp_count,baseline_count,channel_count,
                        	                     	      row_count,reference_wavelengths,field_array,
							      imaging_field);
		printf(" <DONE>\n");	
    } else { //enable faceting
	      size_t no_facet_pixels = facet_nx*facet_ny;
	      for (size_t facet_index = 0; facet_index < num_facet_centres; ++facet_index){
			uvw_base_type new_phase_ra = facet_centres[2*facet_index];
                        uvw_base_type new_phase_dec = facet_centres[2*facet_index + 1];

                        printf("FACETING (%f,%f,%f,%f) %lu / %lu...",phase_centre_ra,phase_centre_dec,new_phase_ra,new_phase_dec,facet_index+1, num_facet_centres);
                        fflush(stdout);
	
			typedef imaging::baseline_transform_policy<uvw_base_type, 
								   transform_facet_righthanded_ra_dec> baseline_transform_policy_type;
			typedef imaging::phase_transform_policy<visibility_base_type, 
								uvw_base_type, 
								transform_enable_phase_rotation_righthanded_ra_dec> phase_transform_policy_type;
			typedef imaging::polarization_gridding_policy<visibility_base_type, uvw_base_type, 
								      visibility_weights_base_type, convolution_base_type, grid_base_type,
								      phase_transform_policy_type, gridding_4_pol> polarization_gridding_policy_type;
			typedef imaging::convolution_policy<convolution_base_type,uvw_base_type,
							    polarization_gridding_policy_type,convolution_precomputed_fir> convolution_policy_type;
			
			baseline_transform_policy_type uvw_transform(0,0,casa::Quantity(phase_centre_ra,"arcsec"),casa::Quantity(phase_centre_dec,"arcsec"),
								     casa::Quantity(new_phase_ra,"arcsec"),casa::Quantity(new_phase_dec,"arcsec")); //lm faceting
			//baseline_transform_policy_type uvw_transform; //uv faceting
			phase_transform_policy_type phase_transform(casa::Quantity(phase_centre_ra,"arcsec"),casa::Quantity(phase_centre_dec,"arcsec"),
								    casa::Quantity(new_phase_ra,"arcsec"),casa::Quantity(new_phase_dec,"arcsec")); //lm faceting
			
			polarization_gridding_policy_type polarization_policy(phase_transform,
									      output_buffer + no_facet_pixels*facet_index*number_of_polarization_terms,
									      visibilities,
									      visibility_weights,
									      flags);
			convolution_policy_type convolution_policy(facet_nx,facet_ny,
								   conv_support,conv_oversample,
								   conv, polarization_policy);
			imaging::grid<visibility_base_type,uvw_base_type,
				      reference_wavelengths_base_type,convolution_base_type,
				      visibility_weights_base_type,grid_base_type,
				      baseline_transform_policy_type,
				      polarization_gridding_policy_type,
				      convolution_policy_type>(polarization_policy,uvw_transform,convolution_policy,
									  uvw_coords,
									  flagged_rows,
									  facet_nx,facet_ny,
									  casa::Quantity(facet_cell_size_x,"arcsec"),casa::Quantity(facet_cell_size_y,"arcsec"),
									  timestamp_count,baseline_count,channel_count,
									  row_count,reference_wavelengths,field_array,
									  imaging_field);
			printf(" <DONE>\n");	
		}
    }
  }
}
