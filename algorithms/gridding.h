#pragma once

#include <cstdio>
#include <cstdint>
#include <complex>
#include <string>
#include <casa/Quanta/Quantum.h>

#include "python2.7/Python.h"
#include "numpy/arrayobject.h"
#include "uvw_coord.h"
#include "baseline_transform_policies.h"
#include "phase_transform_policies.h"
#include "polarization_gridding_policies.h"
#include <cmath>

//Enable this if you want some progress printed out:
#define GRIDDER_PRINT_PROGRESS

namespace imaging {
	/**
		Convolutional gridding algorithm
		Arguements:
		active_polarization_gridding_policy: precreated policy describing how to grid polarization
		active_phase_transform_policy: precreated phase transform policy
		uvw_coord: set of associated uvw coordinates (measured in arcsec, ***NOT*** in wavelength (ie. meters))
		nx,ny: size of the pre-allocated buffer
		cellx,celly: size of the cells (in arcsec)
		timestamp_count, baseline_count, channel_count, polarization_term_count: (integral counts as specified)
		reference_wavelengths: associated wavelength of each channel used to sample visibilities (wavelength = speed_of_light / channel_frequency)
		conv: precomputed convolution FIR of size (conv_support x conv_oversample)^2, flat-indexed
		conv_support, conv_oversample: integral numbers
		polarization_index: index of the polarization correlation term currently being gridded
			PRECONDITIONS:
			1. timestamp_count x baseline_count x channel_count x polarization_term_count <= ||visibilities||
			2. (conv_support x conv_oversample)^2 == ||conv||
	*/
	template  <typename visibility_base_type, typename uvw_base_type, 
		   typename reference_wavelengths_base_type, typename convolution_base_type,
		   typename weights_base_type,
		   typename grid_base_type,
		   typename baseline_transformation_policy_type = baseline_transform_policy<uvw_base_type, transform_disable_facet_rotation>,
		   typename polarization_gridding_policy_type = polarization_gridding_policy<visibility_base_type,uvw_base_type,
											     weights_base_type,
											     convolution_base_type,
											     grid_base_type,
											     phase_transform_policy<visibility_base_type, 
														    uvw_base_type, 
														    transform_disable_phase_rotation>,
											     gridding_4_pol> >
	void grid(polarization_gridding_policy_type active_polarization_gridding_policy,
		  const baseline_transformation_policy_type & __restrict__ active_baseline_transform_policy,
		  const uvw_coord<uvw_base_type> * __restrict__ uvw_coords,
		  std::size_t nx, std::size_t ny, casa::Quantity cellx, casa::Quantity celly,
		  std::size_t timestamp_count, std::size_t baseline_count, std::size_t channel_count,
		  const reference_wavelengths_base_type *__restrict__ reference_wavelengths,
		  const convolution_base_type * __restrict__ conv, std::size_t conv_support, std::size_t conv_oversample)
	{
		/*
		Scale the UVW coords so that we can image only a limited section of sky (same scaling as used in AIPS++ 
		(casarest package) and by implication lwimager).
		*/
                uvw_base_type u_scale=nx*cellx.getValue("rad");
                uvw_base_type v_scale=ny*celly.getValue("rad");
		std::size_t grid_u_centre=nx/2;
                std::size_t grid_v_centre=ny/2;
		std::size_t grid_size_in_pixels=nx*ny;
		std::size_t conv_dim_size = (conv_support * conv_oversample);
		std::size_t conv_dim_centre = conv_dim_size / 2;
		uvw_base_type u_conv_scale = 1/uvw_base_type(conv_oversample); //convolution pixel is oversample times smaller than scaled grid cell size
		uvw_base_type v_conv_scale = 1/uvw_base_type(conv_oversample);
		
		#ifdef GRIDDER_PRINT_PROGRESS
		//give some indication of progress:
		float progress_step_size = 10.0;
		float next_progress_step = progress_step_size;
		#endif
                for (std::size_t bt = 0; bt < baseline_count*timestamp_count; ++bt){
			#ifdef GRIDDER_PRINT_PROGRESS
			float progress = bt/float(baseline_count*timestamp_count)*100.0f;
			if (progress > next_progress_step){
				printf("%f%%... ",next_progress_step);
				fflush(stdout);
				next_progress_step += progress_step_size;
			}
			#endif
                        for (std::size_t c = 0; c < channel_count; ++c){
				/*
				 Get uvw coords and measure the uvw coordinates in terms of wavelength
				*/
                                uvw_coord<uvw_base_type> uvw = uvw_coords[bt];

				uvw._u *= (1/reference_wavelengths[c]);
				uvw._v *= (1/reference_wavelengths[c]);
				
				/*	
				 By default this uvw transform does nothing, but it can be set to do rotate a lw facet to be tangent to a new phase centre		
				 The default phase transform does nothing, but it can be set to rotate the visibilities to a new phase centre in lw / uv faceting.
				 
				 The phase transformation policy is applied to all the polarization terms by the polarization gridding policy. This provides a 
				 neat way of handling multiple polarization options when it comes to gridding. The phase transformation term should be seen 
				 as a scalar Jones term when dealing with 2x2 polarized visibility terms for instance.
				 
				 Refer to Cornwell & Perley (1992), Synthesis Imaging II (1999) and Smirnov I (2011)
				*/
				active_baseline_transform_policy.transform(uvw);
				active_polarization_gridding_policy.transform(bt,c,baseline_count,timestamp_count,channel_count,uvw); //reads and transforms the current visibility
				
				/*
				 Now that all the transformations are done, convert and scale the uv coords down to grid space:
				*/
				uvw._u = uvw._u*u_scale;
				uvw._v = uvw._v*v_scale;
				
				/*
				compute the distance the u,v coordinate is from the bin center, scaled by the oversampling factor
				(size of the jump in imaging space):
				*/
				uvw_base_type frac_u = conv_oversample*(uvw._u - (uvw_base_type)(int64_t)uvw._u); 
				uvw_base_type frac_v = conv_oversample*(uvw._v - (uvw_base_type)(int64_t)uvw._v); 
 				uvw._u += grid_u_centre;
 				uvw._v += grid_v_centre;
				
				/*
				now "convolve" with filter:
				*/
				for (std::size_t conv_v = 0; conv_v < conv_dim_size; ++conv_v){
					std::size_t grid_v = uvw._v + (conv_v - conv_dim_centre)*v_conv_scale;
					if (grid_v >= ny) continue;
					
					std::size_t offset_conv_v = frac_v + conv_v;
					for (std::size_t conv_u = 0; conv_u < conv_dim_size; ++conv_u){
						std::size_t grid_u = uvw._u + (conv_u - conv_dim_centre)*u_conv_scale;
						if (grid_u >= nx) continue;
						
						std::size_t offset_conv_u = frac_u + conv_u;	
						std::size_t conv_flat_index = ((offset_conv_v)*conv_dim_size + offset_conv_u); //flatten convolution index
						//by definition the convolution FIR is 0 outside the support region:
						visibility_base_type conv_weight = (visibility_base_type) ( (offset_conv_u < conv_dim_size && 
													     offset_conv_v < conv_dim_size) ? conv[conv_flat_index] : 0);
						std::size_t grid_flat_index = ((grid_v)*nx+(grid_u)); //flatten grid index
						/*
						 The policy must add the weighted visibility to the correct grid if multiple polarizations are involved:
						 */
						active_polarization_gridding_policy.grid_polarization_terms(grid_flat_index,grid_size_in_pixels,conv_weight);
					}
				}
                        }
                }
                
        }

}
/**
 Python wrapper code:
	module initialization
	grid function wrapper
	methods table
*/
static PyObject *gridding_error;

static PyObject * grid (PyObject *self, PyObject *args) {
	using namespace std;
	using namespace imaging;
	
	//just swap these for doubles if you're passing double precission numpy arrays through!	
	typedef float visibility_base_type;
        typedef float uvw_base_type;
        typedef float reference_wavelengths_base_type;
        typedef float convolution_base_type;
	typedef float polarization_weights_base_type;
	typedef float grid_base_type;
	//Read the python arguements:
	PyArrayObject * visibilities;
	PyArrayObject * uvw_coords;
	PyArrayObject * conv;
	size_t conv_support;
	size_t conv_oversample;
        size_t timestamp_count; 
	size_t baseline_count; 
	size_t channel_count; 
	size_t number_of_polarization_terms;
	size_t polarization_index;
        PyArrayObject * reference_wavelengths;
	PyArrayObject * flags;
	PyArrayObject * polarization_weights;
	double phase_centre_ra;
	double phase_centre_dec;
	PyArrayObject *	facet_centres;
	size_t facet_nx;
	size_t facet_ny;
	double facet_cell_size_x;
	double facet_cell_size_y;
	if (!PyArg_ParseTuple(args, "OOOkkkkkkkOOOddOkkdd", 
			&visibilities,
			&uvw_coords,
		        &conv,
		        &conv_support,
		        &conv_oversample,
			&timestamp_count,
			&baseline_count,
			&channel_count,
			&number_of_polarization_terms,
			&polarization_index,
			&reference_wavelengths,
			&flags,
			&polarization_weights,
			&phase_centre_ra,
			&phase_centre_dec,
			&facet_centres,
			&facet_nx,
			&facet_ny,
			&facet_cell_size_x,
			&facet_cell_size_y
			)){
		PyErr_SetString(gridding_error, "Invalid arguements to grid");
		return nullptr;
	}
	//do some sanity checks:
	{	
		if (uvw_coords->nd != 2) {
			PyErr_SetString(gridding_error, "Invalid number of dimensions on uvw coordinate array, expected 2");
			return nullptr;
		}
		if (uvw_coords->dimensions[1] != 3) {
			PyErr_SetString(gridding_error, "Invalid number of coordinate components in uvw coordinate array, expected 3");
                	return nullptr;
		}
		if (uvw_coords->strides[1] != sizeof(uvw_base_type)){
			PyErr_Format(gridding_error, "Expected %lu-bit float typed uvw array",sizeof(uvw_base_type)*8);
			return nullptr;
		}
		if (reference_wavelengths->nd != 1) {
			PyErr_SetString(gridding_error, "Invalid number of dimensions on channel wavelength array, expected 1");
			return nullptr;
		}
		if (reference_wavelengths->strides[0] != sizeof(reference_wavelengths_base_type)){
			PyErr_Format(gridding_error, "Expected %lu-bit float typed reference wavelength array",sizeof(reference_wavelengths_base_type)*8);
	                return nullptr;
		}
		if (visibilities->nd != 3) {
			PyErr_SetString(gridding_error, "Invalid number of dimensions on visibilities array, expected 3");
                	return nullptr;
		}
		if (visibilities->strides[2] != sizeof(std::complex<visibility_base_type>)) {
                	PyErr_Format(gridding_error, "Expected %lu-bit + %lu-bit complex float typed visibility array",sizeof(visibility_base_type)*8,sizeof(visibility_base_type)*8);
	                return nullptr;
        	}
	        if(conv->nd != 2) {
			PyErr_SetString(gridding_error, "Invalid number of dimensions on convolution filter, expected 2");
			return nullptr;
		}
		if (conv->strides[1] != sizeof(convolution_base_type)){
			PyErr_Format(gridding_error, "Expected %lu-bit float typed convolution array",sizeof(convolution_base_type)*8);
			return nullptr;
		}
		if (flags->nd != 3) {
			PyErr_SetString(gridding_error, "Invalid number of dimensions on flagging array, expected 3");
                	return nullptr;
		}
		if (flags->strides[2] != sizeof(bool)){
			PyErr_SetString(gridding_error, "Expected boolean-typed flags");
	                return nullptr;
		}
		if (polarization_weights->nd != 2) {
        	        PyErr_SetString(gridding_error, "Invalid number of dimensions on visibility weights array, expected 2");
                	return nullptr;
        	}
		if (polarization_weights->strides[1] != sizeof(polarization_weights_base_type)){
        	        PyErr_Format(gridding_error, "Expected %lu-bit float typed visibility weight array",sizeof(polarization_weights_base_type)*8);
                	return nullptr;
        	}
		if ((PyObject*)facet_centres != Py_None){
			if (facet_centres->nd != 2) {
        		        PyErr_SetString(gridding_error, "Invalid number of dimensions on facet centres array, expected 2");
                		return nullptr;
		        }
			if (facet_centres->dimensions[1] != 2){
				PyErr_SetString(gridding_error, "Invalid facet centre list, expected list of (ra,dec) elements");
				return nullptr;
			}
			if (facet_centres->strides[1] != sizeof(uvw_base_type)){
        		        PyErr_Format(gridding_error, "Expected %lu-bit float typed facet centre array",sizeof(uvw_base_type)*8);
                		return nullptr;
		        }
		}
	}
	//switch between normal gridding and facetted gridding:
	if ((PyObject*)facet_centres == Py_None){ //no faceting
		//construct return array
		int dims[3] = {1,(int)facet_nx,(int)facet_ny};
		PyArrayObject * output_grid = (PyArrayObject*) PyArray_FromDims(3, dims, NPY_COMPLEX64);
		size_t no_facet_pixels = facet_nx*facet_ny;
			typedef imaging::baseline_transform_policy<uvw_base_type, 
								   transform_disable_facet_rotation> baseline_transform_policy_type;
			typedef imaging::phase_transform_policy<visibility_base_type, 
								uvw_base_type, 
								transform_disable_phase_rotation> phase_transform_policy_type;
			typedef imaging::polarization_gridding_policy<visibility_base_type, uvw_base_type, 
								      polarization_weights_base_type, convolution_base_type, grid_base_type,
								      phase_transform_policy_type, gridding_single_pol> polarization_gridding_policy_type;
			baseline_transform_policy_type uvw_transform; //standard: no uvw rotation
			phase_transform_policy_type phase_transform; //standard: no phase rotation
			polarization_gridding_policy_type polarization_policy(phase_transform,
									      (std::complex<grid_base_type>*)(output_grid->data),
									      (std::complex<visibility_base_type>*)(visibilities->data),
									      (polarization_weights_base_type*)(polarization_weights->data),
									      (bool*)(flags->data),
									      number_of_polarization_terms,polarization_index);
			imaging::grid<visibility_base_type,uvw_base_type,
				      reference_wavelengths_base_type,convolution_base_type,
				      polarization_weights_base_type,grid_base_type,
				      baseline_transform_policy_type,
				      polarization_gridding_policy_type>
							     (polarization_policy,uvw_transform,
                                        	     	      (uvw_coord<uvw_base_type>*)uvw_coords->data,
	                                             	      facet_nx,facet_ny,
        	                                     	      casa::Quantity(facet_cell_size_x,"arcsec"),casa::Quantity(facet_cell_size_y,"arcsec"),
                	                             	      timestamp_count,baseline_count,channel_count,
                        	                     	      (reference_wavelengths_base_type*)reference_wavelengths->data,
						     	      (convolution_base_type*)conv->data,conv_support,conv_oversample);
		return Py_BuildValue("O",output_grid);	
	} else {
		//construct return array
		int dims[3] = {(int)facet_centres->dimensions[0],(int)facet_nx,(int)facet_ny};
		PyArrayObject * output_grid = (PyArrayObject*) PyArray_FromDims(3, dims, NPY_COMPLEX64);
		size_t no_facet_pixels = facet_nx*facet_ny;
	
		for (size_t facet_index = 0; facet_index < facet_centres->dimensions[0]; ++facet_index){
			printf("FACETING %lu / %lu...",facet_index+1, facet_centres->dimensions[0]);
			fflush(stdout);
			uvw_base_type * facet_centres_data = (uvw_base_type *) (facet_centres->data);
			uvw_base_type new_phase_ra = facet_centres_data[2*facet_index];
			uvw_base_type new_phase_dec = facet_centres_data[2*facet_index + 1];
			
			typedef imaging::baseline_transform_policy<uvw_base_type, 
								   transform_facet_lefthanded_ra_dec> baseline_transform_policy_type;
			typedef imaging::phase_transform_policy<visibility_base_type, 
								uvw_base_type, 
								transform_enable_phase_rotation_lefthanded_ra_dec> phase_transform_policy_type;
			typedef imaging::polarization_gridding_policy<visibility_base_type, uvw_base_type, 
								      polarization_weights_base_type, convolution_base_type, grid_base_type,
								      phase_transform_policy_type, gridding_single_pol> polarization_gridding_policy_type;
			baseline_transform_policy_type uvw_transform(0,0,casa::Quantity(phase_centre_ra,"arcsec"),casa::Quantity(phase_centre_dec,"arcsec"),
								     casa::Quantity(new_phase_ra,"arcsec"),casa::Quantity(new_phase_dec,"arcsec")); //lm faceting
			phase_transform_policy_type phase_transform(casa::Quantity(phase_centre_ra,"arcsec"),casa::Quantity(phase_centre_dec,"arcsec"),
								    casa::Quantity(new_phase_ra,"arcsec"),casa::Quantity(new_phase_dec,"arcsec")); //lm faceting
			
			polarization_gridding_policy_type polarization_policy(phase_transform,
									      (std::complex<grid_base_type>*)(output_grid->data) + no_facet_pixels*facet_index*number_of_polarization_terms,
									      (std::complex<visibility_base_type>*)(visibilities->data),
									      (polarization_weights_base_type*)(polarization_weights->data),
									      (bool*)(flags->data),
									      number_of_polarization_terms,polarization_index);
							      
			imaging::grid<visibility_base_type,uvw_base_type,
				      reference_wavelengths_base_type,convolution_base_type,
				      polarization_weights_base_type,grid_base_type,
				      baseline_transform_policy_type,
				      polarization_gridding_policy_type>(polarization_policy,uvw_transform,
									  (uvw_coord<uvw_base_type>*)uvw_coords->data,
									  facet_nx,facet_ny,
									  casa::Quantity(facet_cell_size_x,"arcsec"),casa::Quantity(facet_cell_size_y,"arcsec"),
									  timestamp_count,baseline_count,channel_count,
									  (reference_wavelengths_base_type*)reference_wavelengths->data,
									  (convolution_base_type*)conv->data,conv_support,conv_oversample);
			printf(" <DONE>\n");	
		}
	
		return Py_BuildValue("O",output_grid);
	}
}


