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
#include <vector>
#include <numeric>
#include <cstring>

#include "gpu_wrapper.h"
#include "dft.h"
#include "templated_gridder.h"
#include "timer.h"
#include "correlation_gridding_traits.h"
#include "correlation_gridding_policies.h"
#include "baseline_transform_policies.h"
#include "baseline_transform_traits.h"
#include "phase_transform_policies.h"
#include "jones_2x2.h"

#include "fft_and_repacking_routines.h"
#define NO_THREADS_PER_BLOCK_DIM 256

extern "C" {
    utils::timer * gridding_walltime;
    utils::timer * inversion_timer;
    imaging::ifft_machine * fftw_ifft_machine;
    cudaStream_t compute_stream;
    gridding_parameters gpu_params;
    bool initialized = false;
    normalization_base_type * normalization_counts;
    double get_gridding_walltime(){
      return gridding_walltime->duration();
    }
    double get_inversion_walltime() {
      return inversion_timer->duration();
    }
    void gridding_barrier(){
      cudaSafeCall(cudaStreamSynchronize(compute_stream));
    }
    void initLibrary(gridding_parameters & params) {
	if (initialized) return;
	initialized = true;
	cudaDeviceReset(); //ensure the device in a safe state
	printf("---------------------------------------Backend: GPU GRIDDING LIBRARY---------------------------------------\n");
	#ifdef BULLSEYE_SINGLE
	printf("Double precision mode: disabled \n");
	#endif
	#ifdef BULLSEYE_DOUBLE
	printf("Double precision mode: enabled \n");
	#endif
	printf("-----------------------------------------------------------------------------------------------------------\n");
        int num_devices, device;
        cudaGetDeviceCount(&num_devices);
        if (num_devices > 0) {
            //get the argmax{devID}(multiProcessorCounts):
            int max_multiprocessors = 0, max_device = 0;
            for (device = 0; device < num_devices; device++) {
                cudaDeviceProp properties;
                cudaGetDeviceProperties(&properties, device);
                if (max_multiprocessors < properties.multiProcessorCount) {
                    max_multiprocessors = properties.multiProcessorCount;
                    max_device = device;
                }
            }
            cudaSetDevice(max_device); //select device
            cudaDeviceReset(); //ensure device is in a safe state before we begin processing

            //print some stats:
            cudaDeviceProp properties;
            cudaGetDeviceProperties(&properties, max_device);

            size_t mem_tot = 0;
            size_t mem_free = 0;
            cudaMemGetInfo  (&mem_free, & mem_tot);
            
            printf("%s, device %d on PCI Bus #%d, clocked at %f GHz\n",properties.name,properties.pciDeviceID,
                   properties.pciBusID,properties.clockRate / 1000000.0);
            printf("Compute capability %d.%d with %f GiB global memory (%f GiB free)\n",properties.major,
                   properties.minor,mem_tot/1024.0/1024.0/1024.0,mem_free/1024.0/1024.0/1024.0);
            printf("%d SMs are available\n",properties.multiProcessorCount);
            printf("-----------------------------------------------------------------------------------------------------------\n");
        } else 
            throw std::runtime_error("Cannot find suitable GPU device. Giving up");
	cudaSafeCall(cudaStreamCreateWithFlags(&compute_stream,cudaStreamNonBlocking));
	gridding_walltime = new utils::timer(compute_stream);
	inversion_timer = new utils::timer(compute_stream);
	fftw_ifft_machine = new imaging::ifft_machine(params);
	//alloc memory for all the arrays on the gpu at the beginning of execution...
	gpu_params = params;
	cudaSafeCall(cudaMalloc((void**)&gpu_params.visibilities, sizeof(std::complex<visibility_base_type>) * params.chunk_max_row_count*params.channel_count*params.number_of_polarization_terms_being_gridded));
	cudaSafeCall(cudaMalloc((void**)&gpu_params.uvw_coords, sizeof(imaging::uvw_coord<uvw_base_type>) * params.chunk_max_row_count));
	cudaSafeCall(cudaMalloc((void**)&gpu_params.reference_wavelengths, sizeof(reference_wavelengths_base_type) * params.channel_count * params.spw_count));
	cudaSafeCall(cudaMemcpy(gpu_params.reference_wavelengths,params.reference_wavelengths,sizeof(reference_wavelengths_base_type) * params.channel_count * params.spw_count,cudaMemcpyHostToDevice));
	cudaSafeCall(cudaMalloc((void**)&gpu_params.enabled_channels, sizeof(bool) * params.channel_count * params.spw_count));
	cudaSafeCall(cudaMemcpy(gpu_params.enabled_channels,params.enabled_channels, sizeof(bool) * params.channel_count * params.spw_count,cudaMemcpyHostToDevice));
	cudaSafeCall(cudaMalloc((void**)&gpu_params.channel_grid_indicies, sizeof(size_t) * params.channel_count * params.spw_count));
	cudaSafeCall(cudaMemcpy(gpu_params.channel_grid_indicies,params.channel_grid_indicies, sizeof(size_t) * params.channel_count * params.spw_count,cudaMemcpyHostToDevice));
	if (params.should_grid_sampling_function){
	  cudaSafeCall(cudaMalloc((void**)&gpu_params.sampling_function_channel_grid_indicies, sizeof(size_t) * params.channel_count * params.spw_count))
	  cudaSafeCall(cudaMemcpy(gpu_params.sampling_function_channel_grid_indicies,
				  params.sampling_function_channel_grid_indicies, 
				  sizeof(size_t) * params.channel_count * params.spw_count,cudaMemcpyHostToDevice));
	}
	cudaSafeCall(cudaMalloc((void**)&gpu_params.spw_index_array, sizeof(unsigned int) * params.chunk_max_row_count));
	cudaSafeCall(cudaMalloc((void**)&gpu_params.flagged_rows, sizeof(bool) * params.chunk_max_row_count));
	cudaSafeCall(cudaMalloc((void**)&gpu_params.field_array, sizeof(unsigned int) * params.chunk_max_row_count));
	cudaSafeCall(cudaMalloc((void**)&gpu_params.visibility_weights, sizeof(visibility_weights_base_type) * params.chunk_max_row_count * params.channel_count * params.number_of_polarization_terms_being_gridded));
	cudaSafeCall(cudaMalloc((void**)&gpu_params.flags, sizeof(bool) * params.chunk_max_row_count * params.channel_count * params.number_of_polarization_terms_being_gridded));
	cudaSafeCall(cudaMalloc((void**)&gpu_params.output_buffer, sizeof(std::complex<grid_base_type>) * params.nx * params.ny * 
								    params.number_of_polarization_terms_being_gridded * params.cube_channel_dim_size * params.num_facet_centres));
	cudaSafeCall(cudaMemcpy(gpu_params.output_buffer,params.output_buffer,sizeof(std::complex<grid_base_type>) * params.nx * params.ny * 
				params.number_of_polarization_terms_being_gridded * params.cube_channel_dim_size * params.num_facet_centres,cudaMemcpyHostToDevice));
	if (params.should_grid_sampling_function){
	  cudaSafeCall(cudaMalloc((void**)&gpu_params.sampling_function_buffer, sizeof(std::complex<grid_base_type>) * params.nx * params.ny * 
							params.sampling_function_channel_count * params.num_facet_centres));
	  cudaSafeCall(cudaMemcpy(gpu_params.sampling_function_buffer,params.sampling_function_buffer,sizeof(std::complex<grid_base_type>) * params.nx * params.ny * 
						    params.sampling_function_channel_count * params.num_facet_centres,cudaMemcpyHostToDevice));
	}
	//the filter looks something like |...|...|...|...| where the last oversample worth of taps at both ends are the necessary extra samples 
	//required for improved interpolation in the gridding
	size_t padded_full_support = params.conv_support * 2 + 1 + 2;
	size_t size_of_convolution_function = (padded_full_support) + (padded_full_support - 1) * (params.conv_oversample - 1);
	cudaSafeCall(cudaMalloc((void**)&gpu_params.conv, sizeof(convolution_base_type) * size_of_convolution_function));
	cudaSafeCall(cudaMemcpy(gpu_params.conv, params.conv, sizeof(convolution_base_type) * size_of_convolution_function,cudaMemcpyHostToDevice));
	cudaSafeCall(cudaMalloc((void**)&gpu_params.baseline_starting_indexes, sizeof(size_t) * (params.baseline_count+1)));
	cudaSafeCall(cudaMalloc((void**)&gpu_params.facet_centres, sizeof(uvw_base_type) * params.num_facet_centres * 2)); //enough space to store ra,dec coordinates of facet delay centres
	cudaSafeCall(cudaMemcpy(gpu_params.facet_centres,params.facet_centres, sizeof(uvw_base_type) * params.num_facet_centres * 2,cudaMemcpyHostToDevice));
	if (gpu_params.should_invert_jones_terms){
	  cudaSafeCall(cudaMalloc((void**)&gpu_params.antenna_1_ids, sizeof(unsigned int) * (params.chunk_max_row_count)));
	  cudaSafeCall(cudaMalloc((void**)&gpu_params.antenna_2_ids, sizeof(unsigned int) * (params.chunk_max_row_count)));
	  cudaSafeCall(cudaMalloc((void**)&gpu_params.timestamp_ids, sizeof(size_t) * (params.chunk_max_row_count)));
	  size_t no_timesteps_needed = (params.chunk_max_row_count / params.baseline_count + 1);
	  cudaSafeCall(cudaMalloc((void**)&gpu_params.jones_terms,sizeof(imaging::jones_2x2<visibility_base_type>) * (no_timesteps_needed *
												  params.antenna_count *
												  params.num_facet_centres * 
												  params.spw_count * 
												  params.channel_count)));
	  cudaSafeCall(cudaMalloc((void**)&gpu_params.antenna_jones_starting_indexes,sizeof(size_t) * (params.antenna_count + 1)));
	  cudaSafeCall(cudaMalloc((void**)&gpu_params.jones_time_indicies_per_antenna,sizeof(size_t) * params.antenna_count * no_timesteps_needed));
	}
	size_t no_reduction_bins = (params.conv_support * 2 + 1) * (params.conv_support * 2 + 1);
	normalization_counts = new normalization_base_type[params.num_facet_centres * params.cube_channel_dim_size * params.number_of_polarization_terms_being_gridded * no_reduction_bins]();
	params.normalization_terms = normalization_counts;
	cudaSafeCall(cudaMalloc((void**)&gpu_params.normalization_terms,sizeof(normalization_base_type) * params.num_facet_centres * 
									params.cube_channel_dim_size * params.number_of_polarization_terms_being_gridded * 
									no_reduction_bins));
	cudaSafeCall(cudaMemset(gpu_params.normalization_terms,0,sizeof(normalization_base_type) * params.num_facet_centres * 
									params.cube_channel_dim_size * params.number_of_polarization_terms_being_gridded * 
									no_reduction_bins));
	cudaSafeCall(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));
    }
    void releaseLibrary() {
      if (!initialized) return;
      initialized = false;
      cudaDeviceSynchronize();
      cudaSafeCall(cudaFree(gpu_params.output_buffer));
      if (gpu_params.should_grid_sampling_function)
	cudaSafeCall(cudaFree(gpu_params.sampling_function_buffer));
      cudaSafeCall(cudaFree(gpu_params.visibilities));
      cudaSafeCall(cudaFree(gpu_params.uvw_coords));
      cudaSafeCall(cudaFree(gpu_params.reference_wavelengths));
      cudaSafeCall(cudaFree(gpu_params.enabled_channels));
      cudaSafeCall(cudaFree(gpu_params.channel_grid_indicies));
      if (gpu_params.should_grid_sampling_function)
	cudaSafeCall(cudaFree(gpu_params.sampling_function_channel_grid_indicies));
      cudaSafeCall(cudaFree(gpu_params.spw_index_array));
      cudaSafeCall(cudaFree(gpu_params.flagged_rows));
      cudaSafeCall(cudaFree(gpu_params.field_array));
      cudaSafeCall(cudaFree(gpu_params.flags));
      cudaSafeCall(cudaFree(gpu_params.conv));
      cudaSafeCall(cudaFree(gpu_params.baseline_starting_indexes));
      cudaSafeCall(cudaFree(gpu_params.facet_centres));
      if (gpu_params.should_invert_jones_terms){
	cudaSafeCall(cudaFree(gpu_params.antenna_1_ids));
	cudaSafeCall(cudaFree(gpu_params.antenna_2_ids));
	cudaSafeCall(cudaFree(gpu_params.timestamp_ids));
	cudaSafeCall(cudaFree(gpu_params.jones_terms));
	cudaSafeCall(cudaFree(gpu_params.antenna_jones_starting_indexes));
	cudaSafeCall(cudaFree(gpu_params.jones_time_indicies_per_antenna));
      }
      cudaSafeCall(cudaFree(gpu_params.normalization_terms));
      cudaSafeCall(cudaStreamDestroy(compute_stream));
      delete gridding_walltime;
      delete inversion_timer;
      delete fftw_ifft_machine;
      delete[] normalization_counts;
      cudaDeviceReset(); //leave the device in a safe state
    }
    void weight_uniformly(gridding_parameters & params){
      #define EPSILON 0.0000001f
      gridding_barrier();
      for (std::size_t f = 0; f < params.num_facet_centres; ++f)
	for (std::size_t g = 0; g < params.cube_channel_dim_size; ++g)
	  for (std::size_t y = 0; y < params.ny; ++y)
	    for (std::size_t x = 0; x < params.nx; ++x){
		grid_base_type count = EPSILON;
		//accumulate all the sampling functions that contribute to the current grid
		for (std::size_t c = 0; c < params.sampling_function_channel_count; ++c)
		    count += (int)(params.channel_grid_indicies[c] == g) *
			     real(params.sampling_function_buffer[((f*params.sampling_function_channel_count + c)*params.ny+y)*params.nx + x]);
		count = 1/count;
		//and apply to the continuous block of nx*ny*cube_channel grids (any temporary correlation term buffers should have been collapsed by this point)
		for (size_t corr = 0; corr < params.number_of_polarization_terms_being_gridded; ++corr)
		  params.output_buffer[(((f*params.cube_channel_dim_size + g)*params.number_of_polarization_terms_being_gridded+corr)*params.ny+y)*params.nx+x] *= count;
	    }
    }
    void copy_back_grid_if_last_stamp(gridding_parameters & params,const gridding_parameters & gpu_params){
      if (params.is_final_data_chunk){
	gridding_barrier();
	cudaSafeCall(cudaMemcpy(params.output_buffer,gpu_params.output_buffer,sizeof(std::complex<grid_base_type>) * params.nx * params.ny * 
				params.number_of_polarization_terms_being_gridded * params.cube_channel_dim_size * params.num_facet_centres,
				cudaMemcpyDeviceToHost));
	size_t no_reduction_bins = (params.conv_support * 2 + 1) * (params.conv_support * 2 + 1);
	cudaSafeCall(cudaMemcpy(params.normalization_terms,gpu_params.normalization_terms,
				sizeof(normalization_base_type) * params.num_facet_centres * 
				  params.cube_channel_dim_size * params.number_of_polarization_terms_being_gridded * 
				  no_reduction_bins,
				cudaMemcpyDeviceToHost));
      }  
    }
    void copy_back_sampling_function_if_last_stamp(gridding_parameters & params,const gridding_parameters & gpu_params){
      if (params.is_final_data_chunk){
	gridding_barrier();
	cudaSafeCall(cudaMemcpy(params.sampling_function_buffer,gpu_params.sampling_function_buffer,sizeof(std::complex<grid_base_type>) * params.nx * params.ny * 
				params.sampling_function_channel_count * params.num_facet_centres,
				cudaMemcpyDeviceToHost));
      }  
    }
    void normalize(gridding_parameters & params) {
      gridding_barrier();
      size_t no_reduction_bins = (params.conv_support * 2 + 1) * (params.conv_support * 2 + 1);
      /*
       * Now normalize per facet, per channel accumulator grid and correlation
       */
      for (size_t f = 0; f < params.num_facet_centres; ++f){
	 for (size_t ch = 0; ch < params.cube_channel_dim_size; ++ch){
	    for (size_t corr = 0; corr < params.number_of_polarization_terms_being_gridded; ++corr){
	      normalization_base_type norm_val = 0;
	      for (size_t thid = 0; thid < no_reduction_bins; ++thid){
		norm_val += params.normalization_terms[((f * params.cube_channel_dim_size + ch) * 
							params.number_of_polarization_terms_being_gridded + corr) * no_reduction_bins + thid];
	      }
	      printf("Normalizing cube slice @ facet %lu, channel slice %lu, correlation term %lu with val %f\n",
		     f,ch,corr,norm_val);
	      std::complex<grid_base_type> * __restrict__ grid_ptr = params.output_buffer + 
								     ((f * params.cube_channel_dim_size + ch) * 
								     params.number_of_polarization_terms_being_gridded + corr) * params.ny * params.nx;
	      for (size_t i = 0; i < params.ny * params.nx; ++i)
		grid_ptr[i] /= norm_val;
	    }
	  }
	}
    }
    void finalize(gridding_parameters & params) {
        gridding_barrier();
	cudaStream_t inversion_timing_stream;
	cudaSafeCall(cudaStreamCreateWithFlags(&inversion_timing_stream,cudaStreamNonBlocking));
	inversion_timer->start();
        fftw_ifft_machine->repack_and_ifft_uv_grids(params);
	inversion_timer->stop();
	cudaSafeCall(cudaStreamDestroy(inversion_timing_stream));
    }
    void finalize_psf(gridding_parameters & params) {
        gridding_barrier();
	cudaStream_t inversion_timing_stream;
	cudaSafeCall(cudaStreamCreateWithFlags(&inversion_timing_stream,cudaStreamNonBlocking));
	inversion_timer->start();
	fftw_ifft_machine->repack_and_ifft_sampling_function_grids(params);
	inversion_timer->stop();
	cudaSafeCall(cudaStreamDestroy(inversion_timing_stream));
    }
    long compute_baseline_index(long a1, long a2, long no_antennae){
      //There is a quadratic series expression relating a1 and a2 to a unique baseline index (can be found by the double difference method)
      //Let slow varying index be S = min(a1,a2)
      //The goal is to find the number of fast varying terms (as the slow varying terms increase these get fewer and fewer, because we
      //only consider unique baselines and not the negative baselines)
      //B = (-S^2 + 2*S*#Ant + S) / 2 + diff between the slowest and fastest varying antenna
      long slow_changing_antenna_index = std::min(a1,a2);
      return (slow_changing_antenna_index*(-slow_changing_antenna_index + (2*no_antennae + 1))) / 2 + std::abs(a1 - a2);
    }
    void repack_input_data(gridding_parameters & params){
	//Romein's gridding strategy requires that we repack things per baseline
	using namespace std;
	using namespace imaging;
	//this expects ||params.baseline_staring_indexes|| == N*(N-1)/2 + N + 1 (because we want to implicitly encode the size of the last baseline)
	//This section will compute the prescan operation over addition
	{
	  memset((void*)params.baseline_starting_indexes,0,sizeof(size_t)*(params.baseline_count + 1));
	  for (size_t r = 0; r < params.row_count; ++r){
	    size_t bi = compute_baseline_index(params.antenna_1_ids[r],params.antenna_2_ids[r],params.antenna_count);
	    ++params.baseline_starting_indexes[bi+1];
	  }
	  partial_sum(params.baseline_starting_indexes,
		      params.baseline_starting_indexes + params.baseline_count + 1,
		      params.baseline_starting_indexes);
	}
	//Now alloc temp storage and run through the rows again to copy the data per baseline
	{
	  //these are all sizes of each of the columns specified in the NRAO Measurement Set v2.0 definition (http://casa.nrao.edu/Memos/229.html)
	  vector<uvw_coord<uvw_base_type> > tmp_uvw(params.row_count,0);
	  vector<std::complex<visibility_base_type> > tmp_data(params.row_count*params.channel_count*params.number_of_polarization_terms,0);
	  vector<visibility_weights_base_type> tmp_weights(params.row_count*params.channel_count*params.number_of_polarization_terms,1);
	  typedef uint8_t std_bool; //something is wrong with nv_bool... cant get the address of the damn thing so lets just copy bytes
	  vector<std_bool> tmp_flags(params.row_count*params.channel_count*params.number_of_polarization_terms,0);
	  vector<std_bool> tmp_flag_rows(params.row_count,0);
	  vector<unsigned int> tmp_data_desc(params.row_count,0);
	  vector<unsigned int> tmp_ant_1(params.row_count,0);
	  vector<unsigned int> tmp_ant_2(params.row_count,0);
	  vector<unsigned int> tmp_field(params.row_count,0);
	  vector<size_t> tmp_time(params.row_count,0);
	  vector<size_t> current_baseline_timestamp_index(params.baseline_count,0);
	  //reorder per baseline
	  for (size_t r = 0; r < params.row_count; ++r){
		size_t bi = compute_baseline_index(params.antenna_1_ids[r],params.antenna_2_ids[r],params.antenna_count);
		size_t rearanged_index = current_baseline_timestamp_index[bi] + params.baseline_starting_indexes[bi];
		++current_baseline_timestamp_index[bi];
		tmp_uvw[rearanged_index] = ((uvw_coord<uvw_base_type> *)(params.uvw_coords))[r];
		memcpy((void*)(&tmp_data[rearanged_index*params.channel_count*params.number_of_polarization_terms]),
		       (void*)(params.visibilities + r*params.channel_count*params.number_of_polarization_terms),
		       params.channel_count*params.number_of_polarization_terms * sizeof(std::complex<visibility_base_type>));
		memcpy((void*)(&tmp_weights[rearanged_index*params.channel_count*params.number_of_polarization_terms]),
		       (void*)(params.visibility_weights + r*params.channel_count*params.number_of_polarization_terms),
		       params.channel_count*params.number_of_polarization_terms * sizeof(visibility_weights_base_type));
		memcpy((void*)(&tmp_flags[rearanged_index*params.channel_count*params.number_of_polarization_terms]),
		       (void*)(params.flags + r*params.channel_count*params.number_of_polarization_terms),
		       params.channel_count*params.number_of_polarization_terms * sizeof(std_bool));
		tmp_flag_rows[rearanged_index] = params.flagged_rows[r];
		tmp_data_desc[rearanged_index] = params.spw_index_array[r];
		tmp_ant_1[rearanged_index] = params.antenna_1_ids[r];
		tmp_ant_2[rearanged_index] = params.antenna_2_ids[r];
		tmp_field[rearanged_index] = params.field_array[r];
		if (params.should_invert_jones_terms)
		  tmp_time[rearanged_index] = params.timestamp_ids[r];
	  }
	  //copy back to python variables
	  memcpy((void*)(params.uvw_coords),
		 (void*)(&tmp_uvw[0]),
		 params.row_count * sizeof(uvw_coord<uvw_base_type>));
	  memcpy((void*)(params.visibilities),
		 (void*)(&tmp_data[0]),
		 params.row_count*params.channel_count*params.number_of_polarization_terms * sizeof(std::complex<visibility_base_type>));
	  memcpy((void*)(params.visibility_weights),
		 (void*)(&tmp_weights[0]),
		 params.row_count*params.channel_count*params.number_of_polarization_terms * sizeof(visibility_weights_base_type));
	  memcpy((void*)(params.flags),
		 (void*)(&tmp_flags[0]),
		 params.row_count*params.channel_count*params.number_of_polarization_terms * sizeof(std_bool));
	  memcpy((void*)(params.flagged_rows),
		 (void*)(&tmp_flag_rows[0]),
		 params.row_count * sizeof(std_bool));
	  memcpy((void*)(params.spw_index_array),
		 (void*)(&tmp_data_desc[0]),
		 params.row_count * sizeof(unsigned int));
	  memcpy((void*)(params.antenna_1_ids),
		 (void*)(&tmp_ant_1[0]),
		 params.row_count * sizeof(unsigned int));
	  memcpy((void*)(params.antenna_2_ids),
		 (void*)(&tmp_ant_2[0]),
		 params.row_count * sizeof(unsigned int));
	  memcpy((void*)(params.field_array),
		 (void*)(&tmp_field[0]),
		 params.row_count * sizeof(unsigned int));
	  if (params.should_invert_jones_terms)
	    memcpy((void*)(params.timestamp_ids),
		   (void*)(&tmp_time[0]),
		   params.row_count * sizeof(size_t));
	}
    }
    
    void grid_single_pol(gridding_parameters & params){
      gridding_walltime->start();
      printf("Gridding single correlation on the GPU...\n");    
      //copy everything that changed to the gpu
      {
	gpu_params.row_count = params.row_count;
	gpu_params.no_timestamps_read = params.no_timestamps_read;
	gpu_params.is_final_data_chunk = params.is_final_data_chunk;
	//pack and cpy only the necessary visibilities (it doesn't matter if we mod the array here it is not being used again afterwards
	size_t ubound = params.row_count*params.channel_count;
	cudaSafeCall(cudaHostRegister(params.visibilities,sizeof(std::complex<visibility_base_type>) * params.row_count * params.channel_count,0));
	cudaSafeCall(cudaHostRegister(params.spw_index_array,sizeof(unsigned int) * params.row_count,0));
	cudaSafeCall(cudaHostRegister(params.uvw_coords,sizeof(imaging::uvw_coord<uvw_base_type>) * params.row_count,0));
	cudaSafeCall(cudaHostRegister(params.visibility_weights,sizeof(visibility_weights_base_type) * params.row_count * params.channel_count,0));
	cudaSafeCall(cudaHostRegister(params.flagged_rows,sizeof(bool) * params.row_count,0));
	cudaSafeCall(cudaHostRegister(params.flags,sizeof(bool) * params.row_count * params.channel_count,0));
	cudaSafeCall(cudaHostRegister(params.field_array,sizeof(unsigned int) * params.row_count,0));
	cudaSafeCall(cudaHostRegister(params.baseline_starting_indexes, sizeof(size_t) * (params.baseline_count+1),0));
	cudaSafeCall(cudaMemcpyAsync(gpu_params.uvw_coords,params.uvw_coords,sizeof(imaging::uvw_coord<uvw_base_type>) * params.row_count,cudaMemcpyHostToDevice,compute_stream));
	cudaSafeCall(cudaMemcpyAsync(gpu_params.baseline_starting_indexes, params.baseline_starting_indexes, sizeof(size_t) * (params.baseline_count+1),cudaMemcpyHostToDevice,compute_stream));
	cudaSafeCall(cudaMemcpyAsync(gpu_params.field_array,params.field_array,sizeof(unsigned int) * params.row_count,cudaMemcpyHostToDevice,compute_stream));
	cudaSafeCall(cudaMemcpyAsync(gpu_params.flagged_rows,params.flagged_rows,sizeof(bool) * params.row_count,
				     cudaMemcpyHostToDevice,compute_stream));
	cudaSafeCall(cudaMemcpyAsync(gpu_params.spw_index_array,params.spw_index_array,sizeof(unsigned int) * params.row_count,
				     cudaMemcpyHostToDevice,compute_stream));
	//Do not parallelize this:
	for (std::size_t i = 0; i < ubound; ++i){
	    size_t r = i / params.channel_count;
	    size_t c = i - r * params.channel_count;
	    size_t compact_index = r*params.channel_count + c;
	    size_t strided_index = (compact_index)*params.number_of_polarization_terms + params.polarization_index;
	    params.visibilities[compact_index] = params.visibilities[strided_index];
	    params.visibility_weights[compact_index] = params.visibility_weights[strided_index];
	    params.flags[compact_index] = params.flags[strided_index];
	}
	
	cudaSafeCall(cudaMemcpyAsync(gpu_params.visibilities,params.visibilities,sizeof(std::complex<visibility_base_type>) * params.row_count * params.channel_count,
				     cudaMemcpyHostToDevice,compute_stream));
	cudaSafeCall(cudaMemcpyAsync(gpu_params.visibility_weights,params.visibility_weights,sizeof(visibility_weights_base_type) * params.row_count * params.channel_count,
				     cudaMemcpyHostToDevice,compute_stream));
	cudaSafeCall(cudaMemcpyAsync(gpu_params.flags,params.flags,sizeof(bool) * params.row_count * params.channel_count,
				     cudaMemcpyHostToDevice,compute_stream));
	cudaSafeCall(cudaHostUnregister(params.visibilities));
	cudaSafeCall(cudaHostUnregister(params.spw_index_array));
	cudaSafeCall(cudaHostUnregister(params.uvw_coords));
	cudaSafeCall(cudaHostUnregister(params.visibility_weights));
	cudaSafeCall(cudaHostUnregister(params.flagged_rows));
	cudaSafeCall(cudaHostUnregister(params.flags));
	cudaSafeCall(cudaHostUnregister(params.field_array));
	cudaSafeCall(cudaHostUnregister(params.baseline_starting_indexes));
      }
      //invoke computation
      {
	size_t conv_support_size = (params.conv_support*2+1);
	size_t padded_conv_support_size = (conv_support_size+2);
	size_t min_threads_needed = params.baseline_count * conv_support_size * conv_support_size;
	size_t block_size = NO_THREADS_PER_BLOCK_DIM;
	size_t total_blocks_needed = ceil(min_threads_needed / double(block_size));
	size_t total_blocks_needed_per_dim = total_blocks_needed;
	
	
	dim3 no_threads_per_block(block_size,1,1);
	dim3 no_blocks_per_grid(total_blocks_needed_per_dim,1,1);
	size_t size_of_convolution_function = padded_conv_support_size * params.conv_oversample * sizeof(convolution_base_type); //see algorithms/convolution_policies.h for the reason behind the padding
	typedef imaging::correlation_gridding_policy<imaging::grid_single_correlation> correlation_gridding_policy;
	typedef imaging::baseline_transform_policy<imaging::transform_disable_facet_rotation > baseline_transform_policy;
	typedef imaging::phase_transform_policy<imaging::disable_faceting_phase_shift> phase_transform_policy;
	if (params.wplanes > 1)
	  throw std::runtime_error("GPU gridder currently doesn't support w projection options");
	else
	  imaging::templated_gridder<correlation_gridding_policy,baseline_transform_policy,phase_transform_policy><<<no_blocks_per_grid,no_threads_per_block,size_of_convolution_function,compute_stream>>>(gpu_params);
      }
      //swap buffers device -> host when gridded last chunk
      copy_back_grid_if_last_stamp(params,gpu_params);
      gridding_walltime->stop();
    }
    void facet_single_pol(gridding_parameters & params){
     gridding_walltime->start();
      printf("Faceting single correlation on the GPU...\n");    
      //copy everything that changed to the gpu
      {
	gpu_params.row_count = params.row_count;
	gpu_params.no_timestamps_read = params.no_timestamps_read;
	gpu_params.is_final_data_chunk = params.is_final_data_chunk;
	//pack and cpy only the necessary visibilities (it doesn't matter if we mod the array here it is not being used again afterwards
	size_t ubound = params.row_count*params.channel_count;
	cudaSafeCall(cudaHostRegister(params.visibilities,sizeof(std::complex<visibility_base_type>) * params.row_count * params.channel_count,0));
	cudaSafeCall(cudaHostRegister(params.spw_index_array,sizeof(unsigned int) * params.row_count,0));
	cudaSafeCall(cudaHostRegister(params.uvw_coords,sizeof(imaging::uvw_coord<uvw_base_type>) * params.row_count,0));
	cudaSafeCall(cudaHostRegister(params.visibility_weights,sizeof(visibility_weights_base_type) * params.row_count * params.channel_count,0));
	cudaSafeCall(cudaHostRegister(params.flagged_rows,sizeof(bool) * params.row_count,0));
	cudaSafeCall(cudaHostRegister(params.flags,sizeof(bool) * params.row_count * params.channel_count,0));
	cudaSafeCall(cudaHostRegister(params.field_array,sizeof(unsigned int) * params.row_count,0));
	cudaSafeCall(cudaHostRegister(params.baseline_starting_indexes, sizeof(size_t) * (params.baseline_count+1),0));
	cudaSafeCall(cudaMemcpyAsync(gpu_params.uvw_coords,params.uvw_coords,sizeof(imaging::uvw_coord<uvw_base_type>) * params.row_count,cudaMemcpyHostToDevice,compute_stream));
	cudaSafeCall(cudaMemcpyAsync(gpu_params.baseline_starting_indexes, params.baseline_starting_indexes, sizeof(size_t) * (params.baseline_count+1),cudaMemcpyHostToDevice,compute_stream));
	cudaSafeCall(cudaMemcpyAsync(gpu_params.field_array,params.field_array,sizeof(unsigned int) * params.row_count,cudaMemcpyHostToDevice,compute_stream));
	cudaSafeCall(cudaMemcpyAsync(gpu_params.flagged_rows,params.flagged_rows,sizeof(bool) * params.row_count,
				     cudaMemcpyHostToDevice,compute_stream));
	cudaSafeCall(cudaMemcpyAsync(gpu_params.spw_index_array,params.spw_index_array,sizeof(unsigned int) * params.row_count,
				     cudaMemcpyHostToDevice,compute_stream));

	//Do not parallelize this:
	for (std::size_t i = 0; i < ubound; ++i){
	    size_t r = i / params.channel_count;
	    size_t c = i - r * params.channel_count;
	    size_t compact_index = r*params.channel_count + c;
	    size_t strided_index = (compact_index)*params.number_of_polarization_terms + params.polarization_index;
	    params.visibilities[compact_index] = params.visibilities[strided_index];
	    params.visibility_weights[compact_index] = params.visibility_weights[strided_index];
	    params.flags[compact_index] = params.flags[strided_index];
	}
	
	cudaSafeCall(cudaMemcpyAsync(gpu_params.visibilities,params.visibilities,sizeof(std::complex<visibility_base_type>) * params.row_count * params.channel_count,
				     cudaMemcpyHostToDevice,compute_stream));
	cudaSafeCall(cudaMemcpyAsync(gpu_params.visibility_weights,params.visibility_weights,sizeof(visibility_weights_base_type) * params.row_count * params.channel_count,
				     cudaMemcpyHostToDevice,compute_stream));
	cudaSafeCall(cudaMemcpyAsync(gpu_params.flags,params.flags,sizeof(bool) * params.row_count * params.channel_count,
				     cudaMemcpyHostToDevice,compute_stream));
	
	cudaSafeCall(cudaHostUnregister(params.visibilities));
	cudaSafeCall(cudaHostUnregister(params.spw_index_array));
	cudaSafeCall(cudaHostUnregister(params.uvw_coords));
	cudaSafeCall(cudaHostUnregister(params.visibility_weights));
	cudaSafeCall(cudaHostUnregister(params.flagged_rows));
	cudaSafeCall(cudaHostUnregister(params.flags));
	cudaSafeCall(cudaHostUnregister(params.field_array));
	cudaSafeCall(cudaHostUnregister(params.baseline_starting_indexes));
      }
      //invoke computation
      {
	size_t conv_support_size = (params.conv_support*2+1);
	size_t padded_conv_support_size = (conv_support_size+2);
	size_t min_threads_needed = params.baseline_count * conv_support_size * conv_support_size * params.num_facet_centres;
	size_t block_size = NO_THREADS_PER_BLOCK_DIM;
	size_t total_blocks_needed = ceil(min_threads_needed / double(block_size));
	size_t total_blocks_needed_per_dim = total_blocks_needed;
	
	
	dim3 no_threads_per_block(block_size,1,1);
	dim3 no_blocks_per_grid(total_blocks_needed_per_dim,1,1);
	size_t size_of_convolution_function = padded_conv_support_size * params.conv_oversample * sizeof(convolution_base_type); //see algorithms/convolution_policies.h for the reason behind the padding
	typedef imaging::correlation_gridding_policy<imaging::grid_single_correlation> correlation_gridding_policy;
	typedef imaging::baseline_transform_policy<imaging::transform_planar_approx_with_w > baseline_transform_policy;
	typedef imaging::phase_transform_policy<imaging::enable_faceting_phase_shift> phase_transform_policy;
	if (params.wplanes > 1)
	  throw std::runtime_error("GPU gridder currently doesn't support w projection options");
	else
	  imaging::templated_gridder<correlation_gridding_policy,baseline_transform_policy,phase_transform_policy><<<no_blocks_per_grid,no_threads_per_block,size_of_convolution_function,compute_stream>>>(gpu_params);
      }
      //swap buffers device -> host when gridded last chunk
      copy_back_grid_if_last_stamp(params,gpu_params);
      gridding_walltime->stop();
    }
    void grid_duel_pol(gridding_parameters & params){
      gridding_walltime->start();
      printf("Gridding duel correlation on the GPU...\n");    
      //copy everything that changed to the gpu
      {
	gpu_params.row_count = params.row_count;
	gpu_params.no_timestamps_read = params.no_timestamps_read;
	gpu_params.is_final_data_chunk = params.is_final_data_chunk;
	//pack and cpy only the necessary visibilities (it doesn't matter if we mod the array here it is not being used again afterwards
	cudaSafeCall(cudaHostRegister(params.visibilities,sizeof(std::complex<visibility_base_type>) * params.row_count * params.channel_count * params.number_of_polarization_terms_being_gridded,0));
	cudaSafeCall(cudaHostRegister(params.spw_index_array,sizeof(unsigned int) * params.row_count,0));
	cudaSafeCall(cudaHostRegister(params.uvw_coords,sizeof(imaging::uvw_coord<uvw_base_type>) * params.row_count,0));
	cudaSafeCall(cudaHostRegister(params.visibility_weights,sizeof(visibility_weights_base_type) * params.row_count * params.channel_count  * params.number_of_polarization_terms_being_gridded,0));
	cudaSafeCall(cudaHostRegister(params.flagged_rows,sizeof(bool) * params.row_count,0));
	cudaSafeCall(cudaHostRegister(params.flags,sizeof(bool) * params.row_count * params.channel_count * params.number_of_polarization_terms_being_gridded,0));
	cudaSafeCall(cudaHostRegister(params.field_array,sizeof(unsigned int) * params.row_count,0));
	cudaSafeCall(cudaHostRegister(params.baseline_starting_indexes, sizeof(size_t) * (params.baseline_count+1),0));
	cudaSafeCall(cudaMemcpyAsync(gpu_params.uvw_coords,params.uvw_coords,sizeof(imaging::uvw_coord<uvw_base_type>) * params.row_count,cudaMemcpyHostToDevice,compute_stream));
	cudaSafeCall(cudaMemcpyAsync(gpu_params.baseline_starting_indexes, params.baseline_starting_indexes, sizeof(size_t) * (params.baseline_count+1),cudaMemcpyHostToDevice,compute_stream));
	cudaSafeCall(cudaMemcpyAsync(gpu_params.field_array,params.field_array,sizeof(unsigned int) * params.row_count,cudaMemcpyHostToDevice,compute_stream));
	cudaSafeCall(cudaMemcpyAsync(gpu_params.flagged_rows,params.flagged_rows,sizeof(bool) * params.row_count,
				     cudaMemcpyHostToDevice,compute_stream));
	cudaSafeCall(cudaMemcpyAsync(gpu_params.spw_index_array,params.spw_index_array,sizeof(unsigned int) * params.row_count,
				     cudaMemcpyHostToDevice,compute_stream));
	size_t ubound = params.row_count*params.channel_count;
	//Do not parallelize this:
	for (std::size_t i = 0; i < ubound; ++i){
	    size_t r = i / (params.channel_count);
	    size_t c = i - r * params.channel_count;
	    size_t channel_flat_index = r*params.channel_count + c;
	    size_t strided_index_corr1 = (channel_flat_index)*params.number_of_polarization_terms + params.polarization_index;
	    size_t strided_index_corr2 = (channel_flat_index)*params.number_of_polarization_terms + params.second_polarization_index;
	    size_t compact_index_corr1 = channel_flat_index << 1;
	    size_t compact_index_corr2 = compact_index_corr1 + 1;
	    params.visibilities[compact_index_corr1] = params.visibilities[strided_index_corr1];
	    params.visibility_weights[compact_index_corr1] = params.visibility_weights[strided_index_corr1];
	    params.flags[compact_index_corr1] = params.flags[strided_index_corr1];
	    params.visibilities[compact_index_corr2] = params.visibilities[strided_index_corr2];
	    params.visibility_weights[compact_index_corr2] = params.visibility_weights[strided_index_corr2];
	    params.flags[compact_index_corr2] = params.flags[strided_index_corr2];
	}
	
	cudaSafeCall(cudaMemcpyAsync(gpu_params.visibilities,params.visibilities,sizeof(std::complex<visibility_base_type>) * params.row_count * params.channel_count * params.number_of_polarization_terms_being_gridded,
				     cudaMemcpyHostToDevice,compute_stream));
	cudaSafeCall(cudaMemcpyAsync(gpu_params.visibility_weights,params.visibility_weights,sizeof(visibility_weights_base_type) * params.row_count * params.channel_count * params.number_of_polarization_terms_being_gridded,
				     cudaMemcpyHostToDevice,compute_stream));
	cudaSafeCall(cudaMemcpyAsync(gpu_params.flags,params.flags,sizeof(bool) * params.row_count * params.channel_count * params.number_of_polarization_terms_being_gridded,
				     cudaMemcpyHostToDevice,compute_stream));
	
	cudaSafeCall(cudaHostUnregister(params.visibilities));
	cudaSafeCall(cudaHostUnregister(params.spw_index_array));
	cudaSafeCall(cudaHostUnregister(params.uvw_coords));
	cudaSafeCall(cudaHostUnregister(params.visibility_weights));
	cudaSafeCall(cudaHostUnregister(params.flagged_rows));
	cudaSafeCall(cudaHostUnregister(params.flags));
	cudaSafeCall(cudaHostUnregister(params.field_array));
	cudaSafeCall(cudaHostUnregister(params.baseline_starting_indexes));
	{
	  size_t conv_support_size = (params.conv_support*2+1);
	  size_t padded_conv_support_size = (conv_support_size+2);
	  size_t min_threads_needed = params.baseline_count * conv_support_size * conv_support_size;
	  size_t block_size = NO_THREADS_PER_BLOCK_DIM;
	  size_t total_blocks_needed = ceil(min_threads_needed / double(block_size));
	  size_t total_blocks_needed_per_dim = total_blocks_needed;
	
	  dim3 no_threads_per_block(block_size,1,1);
	  dim3 no_blocks_per_grid(total_blocks_needed_per_dim,1,1);
	  size_t size_of_convolution_function = padded_conv_support_size * params.conv_oversample * sizeof(convolution_base_type); //see algorithms/convolution_policies.h for the reason behind the padding
	  typedef imaging::correlation_gridding_policy<imaging::grid_duel_correlation> correlation_gridding_policy;
	  typedef imaging::baseline_transform_policy<imaging::transform_disable_facet_rotation > baseline_transform_policy;
	  typedef imaging::phase_transform_policy<imaging::disable_faceting_phase_shift> phase_transform_policy;
	  if (params.wplanes > 1)
	    throw std::runtime_error("GPU gridder currently doesn't support w projection options");
	  else
	    imaging::templated_gridder<correlation_gridding_policy,baseline_transform_policy,phase_transform_policy><<<no_blocks_per_grid,no_threads_per_block,size_of_convolution_function,compute_stream>>>(gpu_params);
	}
	//swap buffers device -> host when gridded last chunk
	copy_back_grid_if_last_stamp(params,gpu_params);    
      }
      gridding_walltime->stop();
    }
    void facet_duel_pol(gridding_parameters & params){
      gridding_walltime->start();
      printf("Faceting duel correlation on the GPU...\n");    
      //copy everything that changed to the gpu
      {
	gpu_params.row_count = params.row_count;
	gpu_params.no_timestamps_read = params.no_timestamps_read;
	gpu_params.is_final_data_chunk = params.is_final_data_chunk;
	//pack and cpy only the necessary visibilities (it doesn't matter if we mod the array here it is not being used again afterwards
	cudaSafeCall(cudaHostRegister(params.visibilities,sizeof(std::complex<visibility_base_type>) * params.row_count * params.channel_count * params.number_of_polarization_terms_being_gridded,0));
	cudaSafeCall(cudaHostRegister(params.spw_index_array,sizeof(unsigned int) * params.row_count,0));
	cudaSafeCall(cudaHostRegister(params.uvw_coords,sizeof(imaging::uvw_coord<uvw_base_type>) * params.row_count,0));
	cudaSafeCall(cudaHostRegister(params.visibility_weights,sizeof(visibility_weights_base_type) * params.row_count * params.channel_count  * params.number_of_polarization_terms_being_gridded,0));
	cudaSafeCall(cudaHostRegister(params.flagged_rows,sizeof(bool) * params.row_count,0));
	cudaSafeCall(cudaHostRegister(params.flags,sizeof(bool) * params.row_count * params.channel_count * params.number_of_polarization_terms_being_gridded,0));
	cudaSafeCall(cudaHostRegister(params.field_array,sizeof(unsigned int) * params.row_count,0));
	cudaSafeCall(cudaHostRegister(params.baseline_starting_indexes, sizeof(size_t) * (params.baseline_count+1),0));
	cudaSafeCall(cudaMemcpyAsync(gpu_params.uvw_coords,params.uvw_coords,sizeof(imaging::uvw_coord<uvw_base_type>) * params.row_count,cudaMemcpyHostToDevice,compute_stream));
	cudaSafeCall(cudaMemcpyAsync(gpu_params.baseline_starting_indexes, params.baseline_starting_indexes, sizeof(size_t) * (params.baseline_count+1),cudaMemcpyHostToDevice,compute_stream));
	cudaSafeCall(cudaMemcpyAsync(gpu_params.field_array,params.field_array,sizeof(unsigned int) * params.row_count,cudaMemcpyHostToDevice,compute_stream));
	cudaSafeCall(cudaMemcpyAsync(gpu_params.flagged_rows,params.flagged_rows,sizeof(bool) * params.row_count,
				     cudaMemcpyHostToDevice,compute_stream));
	cudaSafeCall(cudaMemcpyAsync(gpu_params.spw_index_array,params.spw_index_array,sizeof(unsigned int) * params.row_count,
				     cudaMemcpyHostToDevice,compute_stream));
	size_t ubound = params.row_count*params.channel_count;
	//Do not parallelize this:
	for (std::size_t i = 0; i < ubound; ++i){
	    size_t r = i / (params.channel_count);
	    size_t c = i - r * params.channel_count;
	    size_t channel_flat_index = r*params.channel_count + c;
	    size_t strided_index_corr1 = (channel_flat_index)*params.number_of_polarization_terms + params.polarization_index;
	    size_t strided_index_corr2 = (channel_flat_index)*params.number_of_polarization_terms + params.second_polarization_index;
	    size_t compact_index_corr1 = channel_flat_index << 1;
	    size_t compact_index_corr2 = compact_index_corr1 + 1;
	    params.visibilities[compact_index_corr1] = params.visibilities[strided_index_corr1];
	    params.visibility_weights[compact_index_corr1] = params.visibility_weights[strided_index_corr1];
	    params.flags[compact_index_corr1] = params.flags[strided_index_corr1];
	    params.visibilities[compact_index_corr2] = params.visibilities[strided_index_corr2];
	    params.visibility_weights[compact_index_corr2] = params.visibility_weights[strided_index_corr2];
	    params.flags[compact_index_corr2] = params.flags[strided_index_corr2];
	}
	
	cudaSafeCall(cudaMemcpyAsync(gpu_params.visibilities,params.visibilities,sizeof(std::complex<visibility_base_type>) * params.row_count * params.channel_count * params.number_of_polarization_terms_being_gridded,
				     cudaMemcpyHostToDevice,compute_stream));
	cudaSafeCall(cudaMemcpyAsync(gpu_params.visibility_weights,params.visibility_weights,sizeof(visibility_weights_base_type) * params.row_count * params.channel_count * params.number_of_polarization_terms_being_gridded,
				     cudaMemcpyHostToDevice,compute_stream));
	cudaSafeCall(cudaMemcpyAsync(gpu_params.flags,params.flags,sizeof(bool) * params.row_count * params.channel_count * params.number_of_polarization_terms_being_gridded,
				     cudaMemcpyHostToDevice,compute_stream));
	
	cudaSafeCall(cudaHostUnregister(params.visibilities));
	cudaSafeCall(cudaHostUnregister(params.spw_index_array));
	cudaSafeCall(cudaHostUnregister(params.uvw_coords));
	cudaSafeCall(cudaHostUnregister(params.visibility_weights));
	cudaSafeCall(cudaHostUnregister(params.flagged_rows));
	cudaSafeCall(cudaHostUnregister(params.flags));
	cudaSafeCall(cudaHostUnregister(params.field_array));
	cudaSafeCall(cudaHostUnregister(params.baseline_starting_indexes));
	{
	  size_t conv_support_size = (params.conv_support*2+1);
	  size_t padded_conv_support_size = (conv_support_size+2);
	  size_t min_threads_needed = params.baseline_count * conv_support_size * conv_support_size * params.num_facet_centres;
	  size_t block_size = NO_THREADS_PER_BLOCK_DIM;
	  size_t total_blocks_needed = ceil(min_threads_needed / double(block_size));
	  size_t total_blocks_needed_per_dim = total_blocks_needed;
	
	  dim3 no_threads_per_block(block_size,1,1);
	  dim3 no_blocks_per_grid(total_blocks_needed_per_dim,1,1);
	  size_t size_of_convolution_function = padded_conv_support_size * params.conv_oversample * sizeof(convolution_base_type); //see algorithms/convolution_policies.h for the reason behind the padding
	  typedef imaging::correlation_gridding_policy<imaging::grid_duel_correlation> correlation_gridding_policy;
	  typedef imaging::baseline_transform_policy<imaging::transform_planar_approx_with_w > baseline_transform_policy;
	  typedef imaging::phase_transform_policy<imaging::enable_faceting_phase_shift> phase_transform_policy;
	  if (params.wplanes > 1)
	    throw std::runtime_error("GPU gridder currently doesn't support w projection options");
	  else
	    imaging::templated_gridder<correlation_gridding_policy,baseline_transform_policy,phase_transform_policy><<<no_blocks_per_grid,no_threads_per_block,size_of_convolution_function,compute_stream>>>(gpu_params);
	}
	//swap buffers device -> host when gridded last chunk
	copy_back_grid_if_last_stamp(params,gpu_params);    
      }
      gridding_walltime->stop();
    }
    void grid_4_cor(gridding_parameters & params){
      gridding_walltime->start();
      printf("Gridding 4 correlation on the GPU...\n");    
      //copy everything that changed to the gpu
      {
	gpu_params.row_count = params.row_count;
	gpu_params.no_timestamps_read = params.no_timestamps_read;
	gpu_params.is_final_data_chunk = params.is_final_data_chunk;
	//copy the read chunk accross to the GPU
	cudaSafeCall(cudaHostRegister(params.visibilities,sizeof(std::complex<visibility_base_type>) * params.row_count * params.channel_count * params.number_of_polarization_terms_being_gridded,0));
	cudaSafeCall(cudaHostRegister(params.spw_index_array,sizeof(unsigned int) * params.row_count,0));
	cudaSafeCall(cudaHostRegister(params.uvw_coords,sizeof(imaging::uvw_coord<uvw_base_type>) * params.row_count,0));
	cudaSafeCall(cudaHostRegister(params.visibility_weights,sizeof(visibility_weights_base_type) * params.row_count * params.channel_count  * params.number_of_polarization_terms_being_gridded,0));
	cudaSafeCall(cudaHostRegister(params.flagged_rows,sizeof(bool) * params.row_count,0));
	cudaSafeCall(cudaHostRegister(params.flags,sizeof(bool) * params.row_count * params.channel_count * params.number_of_polarization_terms_being_gridded,0));
	cudaSafeCall(cudaHostRegister(params.field_array,sizeof(unsigned int) * params.row_count,0));
	cudaSafeCall(cudaHostRegister(params.baseline_starting_indexes, sizeof(size_t) * (params.baseline_count+1),0));
	cudaSafeCall(cudaMemcpyAsync(gpu_params.uvw_coords,params.uvw_coords,sizeof(imaging::uvw_coord<uvw_base_type>) * params.row_count,cudaMemcpyHostToDevice,compute_stream));
	cudaSafeCall(cudaMemcpyAsync(gpu_params.baseline_starting_indexes, params.baseline_starting_indexes, sizeof(size_t) * (params.baseline_count+1),cudaMemcpyHostToDevice,compute_stream));
	cudaSafeCall(cudaMemcpyAsync(gpu_params.field_array,params.field_array,sizeof(unsigned int) * params.row_count,cudaMemcpyHostToDevice,compute_stream));
	cudaSafeCall(cudaMemcpyAsync(gpu_params.flagged_rows,params.flagged_rows,sizeof(bool) * params.row_count,
				     cudaMemcpyHostToDevice,compute_stream));
	cudaSafeCall(cudaMemcpyAsync(gpu_params.spw_index_array,params.spw_index_array,sizeof(unsigned int) * params.row_count,
				     cudaMemcpyHostToDevice,compute_stream));
	cudaSafeCall(cudaMemcpyAsync(gpu_params.visibilities,params.visibilities,sizeof(std::complex<visibility_base_type>) * params.row_count * params.channel_count * params.number_of_polarization_terms_being_gridded,
				     cudaMemcpyHostToDevice,compute_stream));
	cudaSafeCall(cudaMemcpyAsync(gpu_params.visibility_weights,params.visibility_weights,sizeof(visibility_weights_base_type) * params.row_count * params.channel_count * params.number_of_polarization_terms_being_gridded,
				     cudaMemcpyHostToDevice,compute_stream));
	cudaSafeCall(cudaMemcpyAsync(gpu_params.flags,params.flags,sizeof(bool) * params.row_count * params.channel_count * params.number_of_polarization_terms_being_gridded,
				     cudaMemcpyHostToDevice,compute_stream));
	
	cudaSafeCall(cudaHostUnregister(params.visibilities));
	cudaSafeCall(cudaHostUnregister(params.spw_index_array));
	cudaSafeCall(cudaHostUnregister(params.uvw_coords));
	cudaSafeCall(cudaHostUnregister(params.visibility_weights));
	cudaSafeCall(cudaHostUnregister(params.flagged_rows));
	cudaSafeCall(cudaHostUnregister(params.flags));
	cudaSafeCall(cudaHostUnregister(params.field_array));
	cudaSafeCall(cudaHostUnregister(params.baseline_starting_indexes));
	{
	  size_t conv_support_size = (params.conv_support*2+1);
	  size_t padded_conv_support_size = (conv_support_size+2);
	  size_t min_threads_needed = params.baseline_count * conv_support_size * conv_support_size;
	  size_t block_size = NO_THREADS_PER_BLOCK_DIM;
	  size_t total_blocks_needed = ceil(min_threads_needed / double(block_size));
	  size_t total_blocks_needed_per_dim = total_blocks_needed;
	
	  dim3 no_threads_per_block(block_size,1,1);
	  dim3 no_blocks_per_grid(total_blocks_needed_per_dim,1,1);
	  size_t size_of_convolution_function = padded_conv_support_size * params.conv_oversample * sizeof(convolution_base_type); //see algorithms/convolution_policies.h for the reason behind the padding
	  typedef imaging::correlation_gridding_policy<imaging::grid_4_correlation> correlation_gridding_policy;
	  typedef imaging::baseline_transform_policy<imaging::transform_disable_facet_rotation > baseline_transform_policy;
	  typedef imaging::phase_transform_policy<imaging::disable_faceting_phase_shift> phase_transform_policy;
	  if (params.wplanes > 1)
	    throw std::runtime_error("GPU gridder currently doesn't support w projection options");
	  else
	    imaging::templated_gridder<correlation_gridding_policy,baseline_transform_policy,phase_transform_policy><<<no_blocks_per_grid,no_threads_per_block,size_of_convolution_function,compute_stream>>>(gpu_params);
	}
	//swap buffers device -> host when gridded last chunk
	copy_back_grid_if_last_stamp(params,gpu_params);    
      }
      gridding_walltime->stop();
    }
    void facet_4_cor(gridding_parameters & params){
      gridding_walltime->start();
      printf("Faceting 4 correlation on the GPU...\n");    
      //copy everything that changed to the gpu
      {
	gpu_params.row_count = params.row_count;
	gpu_params.no_timestamps_read = params.no_timestamps_read;
	gpu_params.is_final_data_chunk = params.is_final_data_chunk;
	//copy the read chunk accross to the GPU
	cudaSafeCall(cudaHostRegister(params.visibilities,sizeof(std::complex<visibility_base_type>) * params.row_count * params.channel_count * params.number_of_polarization_terms_being_gridded,0));
	cudaSafeCall(cudaHostRegister(params.spw_index_array,sizeof(unsigned int) * params.row_count,0));
	cudaSafeCall(cudaHostRegister(params.uvw_coords,sizeof(imaging::uvw_coord<uvw_base_type>) * params.row_count,0));
	cudaSafeCall(cudaHostRegister(params.visibility_weights,sizeof(visibility_weights_base_type) * params.row_count * params.channel_count  * params.number_of_polarization_terms_being_gridded,0));
	cudaSafeCall(cudaHostRegister(params.flagged_rows,sizeof(bool) * params.row_count,0));
	cudaSafeCall(cudaHostRegister(params.flags,sizeof(bool) * params.row_count * params.channel_count * params.number_of_polarization_terms_being_gridded,0));
	cudaSafeCall(cudaHostRegister(params.field_array,sizeof(unsigned int) * params.row_count,0));
	cudaSafeCall(cudaHostRegister(params.baseline_starting_indexes, sizeof(size_t) * (params.baseline_count+1),0));
	cudaSafeCall(cudaMemcpyAsync(gpu_params.uvw_coords,params.uvw_coords,sizeof(imaging::uvw_coord<uvw_base_type>) * params.row_count,cudaMemcpyHostToDevice,compute_stream));
	cudaSafeCall(cudaMemcpyAsync(gpu_params.baseline_starting_indexes, params.baseline_starting_indexes, sizeof(size_t) * (params.baseline_count+1),cudaMemcpyHostToDevice,compute_stream));
	cudaSafeCall(cudaMemcpyAsync(gpu_params.field_array,params.field_array,sizeof(unsigned int) * params.row_count,cudaMemcpyHostToDevice,compute_stream));
	cudaSafeCall(cudaMemcpyAsync(gpu_params.flagged_rows,params.flagged_rows,sizeof(bool) * params.row_count,
				     cudaMemcpyHostToDevice,compute_stream));
	cudaSafeCall(cudaMemcpyAsync(gpu_params.spw_index_array,params.spw_index_array,sizeof(unsigned int) * params.row_count,
				     cudaMemcpyHostToDevice,compute_stream));
	cudaSafeCall(cudaMemcpyAsync(gpu_params.visibilities,params.visibilities,sizeof(std::complex<visibility_base_type>) * params.row_count * params.channel_count * params.number_of_polarization_terms_being_gridded,
				     cudaMemcpyHostToDevice,compute_stream));
	cudaSafeCall(cudaMemcpyAsync(gpu_params.visibility_weights,params.visibility_weights,sizeof(visibility_weights_base_type) * params.row_count * params.channel_count * params.number_of_polarization_terms_being_gridded,
				     cudaMemcpyHostToDevice,compute_stream));
	cudaSafeCall(cudaMemcpyAsync(gpu_params.flags,params.flags,sizeof(bool) * params.row_count * params.channel_count * params.number_of_polarization_terms_being_gridded,
				     cudaMemcpyHostToDevice,compute_stream));
	
	cudaSafeCall(cudaHostUnregister(params.visibilities));
	cudaSafeCall(cudaHostUnregister(params.spw_index_array));
	cudaSafeCall(cudaHostUnregister(params.uvw_coords));
	cudaSafeCall(cudaHostUnregister(params.visibility_weights));
	cudaSafeCall(cudaHostUnregister(params.flagged_rows));
	cudaSafeCall(cudaHostUnregister(params.flags));
	cudaSafeCall(cudaHostUnregister(params.field_array));
	cudaSafeCall(cudaHostUnregister(params.baseline_starting_indexes));
	{
	  size_t conv_support_size = (params.conv_support*2+1);
	  size_t padded_conv_support_size = (conv_support_size+2);
	  size_t min_threads_needed = params.baseline_count * conv_support_size * conv_support_size * params.num_facet_centres;
	  size_t block_size = NO_THREADS_PER_BLOCK_DIM;
	  size_t total_blocks_needed = ceil(min_threads_needed / double(block_size));
	  size_t total_blocks_needed_per_dim = total_blocks_needed;
	
	  dim3 no_threads_per_block(block_size,1,1);
	  dim3 no_blocks_per_grid(total_blocks_needed_per_dim,1,1);
	  size_t size_of_convolution_function = padded_conv_support_size * params.conv_oversample * sizeof(convolution_base_type); //see algorithms/convolution_policies.h for the reason behind the padding
	  typedef imaging::correlation_gridding_policy<imaging::grid_4_correlation> correlation_gridding_policy;
	  typedef imaging::baseline_transform_policy<imaging::transform_planar_approx_with_w > baseline_transform_policy;
	  typedef imaging::phase_transform_policy<imaging::enable_faceting_phase_shift> phase_transform_policy;
	  if (params.wplanes > 1)
	    throw std::runtime_error("GPU gridder currently doesn't support w projection options");
	  else
	    imaging::templated_gridder<correlation_gridding_policy,baseline_transform_policy,phase_transform_policy><<<no_blocks_per_grid,no_threads_per_block,size_of_convolution_function,compute_stream>>>(gpu_params);
	}
	//swap buffers device -> host when gridded last chunk
	copy_back_grid_if_last_stamp(params,gpu_params);    
      }
      gridding_walltime->stop();
    }
    void facet_4_cor_corrections(gridding_parameters & params){
      gridding_walltime->start();
      printf("Faceting 4 correlation on the GPU with Jones corrections...\n");    
      //copy everything that changed to the gpu
      {
	gpu_params.row_count = params.row_count;
	gpu_params.no_timestamps_read = params.no_timestamps_read;
	gpu_params.is_final_data_chunk = params.is_final_data_chunk;
	//copy the read chunk accross to the GPU
	cudaSafeCall(cudaHostRegister(params.visibilities,sizeof(std::complex<visibility_base_type>) * params.row_count * params.channel_count * params.number_of_polarization_terms_being_gridded,0));
	cudaSafeCall(cudaHostRegister(params.spw_index_array,sizeof(unsigned int) * params.row_count,0));
	cudaSafeCall(cudaHostRegister(params.uvw_coords,sizeof(imaging::uvw_coord<uvw_base_type>) * params.row_count,0));
	cudaSafeCall(cudaHostRegister(params.visibility_weights,sizeof(visibility_weights_base_type) * params.row_count * params.channel_count  * params.number_of_polarization_terms_being_gridded,0));
	cudaSafeCall(cudaHostRegister(params.flagged_rows,sizeof(bool) * params.row_count,0));
	cudaSafeCall(cudaHostRegister(params.flags,sizeof(bool) * params.row_count * params.channel_count * params.number_of_polarization_terms_being_gridded,0));
	cudaSafeCall(cudaHostRegister(params.field_array,sizeof(unsigned int) * params.row_count,0));
	cudaSafeCall(cudaHostRegister(params.baseline_starting_indexes, sizeof(size_t) * (params.baseline_count+1),0));
	cudaSafeCall(cudaHostRegister(params.antenna_1_ids, sizeof(unsigned int) * (params.row_count), 0));
	cudaSafeCall(cudaHostRegister(params.antenna_2_ids, sizeof(unsigned int) * (params.row_count), 0));
	cudaSafeCall(cudaHostRegister(params.timestamp_ids, sizeof(std::size_t) * (params.row_count), 0));
	cudaSafeCall(cudaMemcpyAsync(gpu_params.uvw_coords,params.uvw_coords,sizeof(imaging::uvw_coord<uvw_base_type>) * params.row_count,cudaMemcpyHostToDevice,compute_stream));
	cudaSafeCall(cudaMemcpyAsync(gpu_params.baseline_starting_indexes, params.baseline_starting_indexes, sizeof(size_t) * (params.baseline_count+1),cudaMemcpyHostToDevice,compute_stream));
	cudaSafeCall(cudaMemcpyAsync(gpu_params.field_array,params.field_array,sizeof(unsigned int) * params.row_count,cudaMemcpyHostToDevice,compute_stream));
	cudaSafeCall(cudaMemcpyAsync(gpu_params.flagged_rows,params.flagged_rows,sizeof(bool) * params.row_count,
				     cudaMemcpyHostToDevice,compute_stream));
	cudaSafeCall(cudaMemcpyAsync(gpu_params.spw_index_array,params.spw_index_array,sizeof(unsigned int) * params.row_count,
				     cudaMemcpyHostToDevice,compute_stream));
	cudaSafeCall(cudaMemcpyAsync(gpu_params.visibilities,params.visibilities,sizeof(std::complex<visibility_base_type>) * params.row_count * params.channel_count * params.number_of_polarization_terms_being_gridded,
				     cudaMemcpyHostToDevice,compute_stream));
	cudaSafeCall(cudaMemcpyAsync(gpu_params.visibility_weights,params.visibility_weights,sizeof(visibility_weights_base_type) * params.row_count * params.channel_count * params.number_of_polarization_terms_being_gridded,
				     cudaMemcpyHostToDevice,compute_stream));
	cudaSafeCall(cudaMemcpyAsync(gpu_params.flags,params.flags,sizeof(bool) * params.row_count * params.channel_count * params.number_of_polarization_terms_being_gridded,
				     cudaMemcpyHostToDevice,compute_stream));
	cudaSafeCall(cudaMemcpyAsync(gpu_params.antenna_1_ids,params.antenna_1_ids,sizeof(unsigned int) * params.row_count,
				     cudaMemcpyHostToDevice,compute_stream));
	cudaSafeCall(cudaMemcpyAsync(gpu_params.antenna_2_ids,params.antenna_2_ids,sizeof(unsigned int) * params.row_count,
				     cudaMemcpyHostToDevice,compute_stream));
	cudaSafeCall(cudaMemcpyAsync(gpu_params.timestamp_ids,params.timestamp_ids,sizeof(std::size_t) * params.row_count,
				     cudaMemcpyHostToDevice,compute_stream));
	//repack the jones terms per antenna so that we don't transfer unnecessary stuff over PCI-e
	{
	  using namespace std;
	  //we want to retrieve the # timestamps for every antenna that means we need (n+1) elements in this array
	  vector<size_t> antenna_timestamp_starting_indexes(params.antenna_count + 1,0);
	  //bin
	  {
	    vector<long> antenna_current_timestamp(params.antenna_count,-1);
	    for(size_t row = 0; row < params.row_count; ++row){
	      if ((long)params.timestamp_ids[row] > antenna_current_timestamp[params.antenna_1_ids[row]]){
		antenna_timestamp_starting_indexes[params.antenna_1_ids[row] + 1] += 1;
		antenna_current_timestamp[params.antenna_1_ids[row]] += 1;
	      }
	      if ((long)params.timestamp_ids[row] > antenna_current_timestamp[params.antenna_2_ids[row]]){
		antenna_timestamp_starting_indexes[params.antenna_2_ids[row] + 1] += 1;
		antenna_current_timestamp[params.antenna_2_ids[row]] += 1;
	      }
	    }
	  }
	  //because we binned at antenna_id + 1 partial_sum will compute the prescan (starting timestamp index per antenna)
	  std::partial_sum(antenna_timestamp_starting_indexes.begin(),
			   antenna_timestamp_starting_indexes.end(),
			   antenna_timestamp_starting_indexes.begin());
	  size_t step_size = params.num_facet_centres * params.spw_count * params.channel_count;
	  vector<imaging::jones_2x2<visibility_base_type> > repacked_data(antenna_timestamp_starting_indexes[(params.antenna_count)] * step_size);
	  vector<std::size_t> repacked_indexes(antenna_timestamp_starting_indexes[(params.antenna_count)]);
	  cudaSafeCall(cudaHostRegister(&repacked_data[0], sizeof(imaging::jones_2x2<visibility_base_type>) * repacked_data.size(), 0));
	  cudaSafeCall(cudaHostRegister(&antenna_timestamp_starting_indexes[0], sizeof(size_t) * antenna_timestamp_starting_indexes.size(), 0));
	  cudaSafeCall(cudaMemcpyAsync(gpu_params.antenna_jones_starting_indexes,&antenna_timestamp_starting_indexes[0],
				       sizeof(size_t) * antenna_timestamp_starting_indexes.size(),cudaMemcpyHostToDevice,compute_stream));
	  cudaSafeCall(cudaHostRegister(&repacked_indexes[0], sizeof(size_t) * repacked_indexes.size(), 0));
	  { //now repack
	    vector<long> antenna_current_timestamp(params.antenna_count,-1);

	    for(size_t row = 0; row < params.row_count; ++row){
	      //copy the first antenna into position
	      if ((long)params.timestamp_ids[row] > antenna_current_timestamp[params.antenna_1_ids[row]]){ //a single antenna may be in multiple baselines... don't recopy
		antenna_current_timestamp[params.antenna_1_ids[row]] += 1;
		size_t old_index_antenna_1 = (params.timestamp_ids[row] * params.antenna_count + params.antenna_1_ids[row]) * 
					      step_size;
		size_t new_index_antenna_1 = (antenna_timestamp_starting_indexes[params.antenna_1_ids[row]] +
					      antenna_current_timestamp[params.antenna_1_ids[row]]) *
					      step_size;
		repacked_indexes[new_index_antenna_1/step_size] = params.timestamp_ids[row];
		imaging::jones_2x2<visibility_base_type> * old_arr = (imaging::jones_2x2<visibility_base_type> *) params.jones_terms;
		memcpy(&repacked_data[0] + new_index_antenna_1,
		       old_arr + old_index_antenna_1,
		       step_size * sizeof(imaging::jones_2x2<visibility_base_type>));
	      }
	      //copy the second antenna into position
	      if ((long)params.timestamp_ids[row] > antenna_current_timestamp[params.antenna_2_ids[row]]){ //a single antenna may be in multiple baselines... don't recopy
		antenna_current_timestamp[params.antenna_2_ids[row]] += 1;
		size_t old_index_antenna_2 = (params.timestamp_ids[row] * params.antenna_count + params.antenna_2_ids[row]) * 
					      step_size;
		size_t new_index_antenna_2 = (antenna_timestamp_starting_indexes[params.antenna_2_ids[row]] +
					      antenna_current_timestamp[params.antenna_2_ids[row]]) *
					      step_size;
		repacked_indexes[new_index_antenna_2/step_size] = params.timestamp_ids[row];
		imaging::jones_2x2<visibility_base_type> * old_arr = (imaging::jones_2x2<visibility_base_type> *) params.jones_terms;
		memcpy(&repacked_data[0] + new_index_antenna_2,
		       old_arr + old_index_antenna_2,
		       step_size * sizeof(imaging::jones_2x2<visibility_base_type>));
	      }
	    }
	  }
	  printf("INVERTING %lu JONES MATRICIES\n",repacked_data.size());
	  invert_all(&repacked_data[0], repacked_data.size());
	  cudaSafeCall(cudaMemcpyAsync(gpu_params.jones_terms,&repacked_data[0],sizeof(imaging::jones_2x2<visibility_base_type>) * repacked_data.size(),
		       cudaMemcpyHostToDevice,compute_stream));
	  cudaSafeCall(cudaMemcpyAsync(gpu_params.jones_time_indicies_per_antenna,&repacked_indexes[0],sizeof(size_t) * repacked_indexes.size(),
		       cudaMemcpyHostToDevice,compute_stream));
	  cudaSafeCall(cudaHostUnregister(&repacked_data[0]));
	  cudaSafeCall(cudaHostUnregister(&antenna_timestamp_starting_indexes[0]));
	  cudaSafeCall(cudaHostUnregister(&repacked_indexes[0]));
	  printf("TRANSFERRED %lu REPACKED JONES MATRICIES TO DEVICE\n",repacked_data.size());
	}
	
	cudaSafeCall(cudaHostUnregister(params.visibilities));
	cudaSafeCall(cudaHostUnregister(params.spw_index_array));
	cudaSafeCall(cudaHostUnregister(params.uvw_coords));
	cudaSafeCall(cudaHostUnregister(params.visibility_weights));
	cudaSafeCall(cudaHostUnregister(params.flagged_rows));
	cudaSafeCall(cudaHostUnregister(params.flags));
	cudaSafeCall(cudaHostUnregister(params.field_array));
	cudaSafeCall(cudaHostUnregister(params.baseline_starting_indexes));
	cudaSafeCall(cudaHostUnregister(params.antenna_1_ids));
	cudaSafeCall(cudaHostUnregister(params.antenna_2_ids));
	cudaSafeCall(cudaHostUnregister(params.timestamp_ids));
	{
	  size_t conv_support_size = (params.conv_support*2+1);
	  size_t padded_conv_support_size = (conv_support_size+2);
	  size_t min_threads_needed = params.baseline_count * conv_support_size * conv_support_size * params.num_facet_centres;
	  size_t block_size = NO_THREADS_PER_BLOCK_DIM;
	  size_t total_blocks_needed = ceil(min_threads_needed / double(block_size));
	  size_t total_blocks_needed_per_dim = total_blocks_needed;
	
	  dim3 no_threads_per_block(block_size,1,1);
	  dim3 no_blocks_per_grid(total_blocks_needed_per_dim,1,1);
	  size_t size_of_convolution_function = padded_conv_support_size * params.conv_oversample * sizeof(convolution_base_type); //see algorithms/convolution_policies.h for the reason behind the padding
	  typedef imaging::correlation_gridding_policy<imaging::grid_4_correlation> correlation_gridding_policy;
	  typedef imaging::baseline_transform_policy<imaging::transform_planar_approx_with_w > baseline_transform_policy;
	  typedef imaging::phase_transform_policy<imaging::enable_faceting_phase_shift> phase_transform_policy;
	  if (params.wplanes > 1)
	    throw std::runtime_error("GPU gridder currently doesn't support w projection options");
	  else
	    imaging::templated_gridder<correlation_gridding_policy,baseline_transform_policy,phase_transform_policy><<<no_blocks_per_grid,no_threads_per_block,size_of_convolution_function,compute_stream>>>(gpu_params);
	}
	//swap buffers device -> host when gridded last chunk
	copy_back_grid_if_last_stamp(params,gpu_params);    
      }
      gridding_walltime->stop();
    }
    void grid_sampling_function(gridding_parameters & params){
      gridding_barrier();
      gridding_walltime->start();
      printf("Gridding sampling function on the GPU...\n");    
      //copy everything that changed to the gpu
      {
	gpu_params.row_count = params.row_count;
	gpu_params.no_timestamps_read = params.no_timestamps_read;
	gpu_params.is_final_data_chunk = params.is_final_data_chunk;
	//pack and cpy only the necessary parameters
	cudaSafeCall(cudaHostRegister(params.spw_index_array,sizeof(unsigned int) * params.row_count,0));
	cudaSafeCall(cudaHostRegister(params.uvw_coords,sizeof(imaging::uvw_coord<uvw_base_type>) * params.row_count,0));
	cudaSafeCall(cudaHostRegister(params.field_array,sizeof(unsigned int) * params.row_count,0));
	cudaSafeCall(cudaHostRegister(params.baseline_starting_indexes, sizeof(size_t) * (params.baseline_count+1),0));
	cudaSafeCall(cudaMemcpyAsync(gpu_params.uvw_coords,params.uvw_coords,sizeof(imaging::uvw_coord<uvw_base_type>) * params.row_count,cudaMemcpyHostToDevice,compute_stream));
	cudaSafeCall(cudaMemcpyAsync(gpu_params.baseline_starting_indexes, params.baseline_starting_indexes, sizeof(size_t) * (params.baseline_count+1),cudaMemcpyHostToDevice,compute_stream));
	cudaSafeCall(cudaMemcpyAsync(gpu_params.field_array,params.field_array,sizeof(unsigned int) * params.row_count,cudaMemcpyHostToDevice,compute_stream));
	cudaSafeCall(cudaMemcpyAsync(gpu_params.spw_index_array,params.spw_index_array,sizeof(unsigned int) * params.row_count,
				     cudaMemcpyHostToDevice,compute_stream));
	
	cudaSafeCall(cudaHostUnregister(params.spw_index_array));
	cudaSafeCall(cudaHostUnregister(params.uvw_coords));
	cudaSafeCall(cudaHostUnregister(params.field_array));
	cudaSafeCall(cudaHostUnregister(params.baseline_starting_indexes));
      }
      //invoke computation
      {
	size_t conv_support_size = (params.conv_support*2+1);
	size_t padded_conv_support_size = (conv_support_size+2);
	size_t min_threads_needed = params.baseline_count * conv_support_size * conv_support_size;
	size_t block_size = NO_THREADS_PER_BLOCK_DIM;
	size_t total_blocks_needed = ceil(min_threads_needed / double(block_size));
	size_t total_blocks_needed_per_dim = total_blocks_needed;
	
	
	dim3 no_threads_per_block(block_size,1,1);
	dim3 no_blocks_per_grid(total_blocks_needed_per_dim,1,1);
	size_t size_of_convolution_function = padded_conv_support_size * params.conv_oversample * sizeof(convolution_base_type); //see algorithms/convolution_policies.h for the reason behind the padding
	typedef imaging::correlation_gridding_policy<imaging::grid_sampling_function> correlation_gridding_policy;
	typedef imaging::baseline_transform_policy<imaging::transform_disable_facet_rotation > baseline_transform_policy;
	typedef imaging::phase_transform_policy<imaging::disable_faceting_phase_shift> phase_transform_policy;
	if (params.wplanes > 1)
	  throw std::runtime_error("GPU gridder currently doesn't support w projection options");
	else
	  imaging::templated_gridder<correlation_gridding_policy,baseline_transform_policy,phase_transform_policy><<<no_blocks_per_grid,no_threads_per_block,size_of_convolution_function,compute_stream>>>(gpu_params);
      }
      //swap buffers device -> host when gridded last chunk
      copy_back_sampling_function_if_last_stamp(params,gpu_params);
      gridding_walltime->stop();
    }
    void facet_sampling_function(gridding_parameters & params){
      gridding_barrier();
      gridding_walltime->start();
      printf("Faceting sampling function on the GPU...\n");    
      //copy everything that changed to the gpu
      {
	gpu_params.row_count = params.row_count;
	gpu_params.no_timestamps_read = params.no_timestamps_read;
	gpu_params.is_final_data_chunk = params.is_final_data_chunk;
	//cpy only the necessary fields
	cudaSafeCall(cudaHostRegister(params.spw_index_array,sizeof(unsigned int) * params.row_count,0));
	cudaSafeCall(cudaHostRegister(params.uvw_coords,sizeof(imaging::uvw_coord<uvw_base_type>) * params.row_count,0));
	cudaSafeCall(cudaHostRegister(params.field_array,sizeof(unsigned int) * params.row_count,0));
	cudaSafeCall(cudaHostRegister(params.baseline_starting_indexes, sizeof(size_t) * (params.baseline_count+1),0));
	cudaSafeCall(cudaMemcpyAsync(gpu_params.uvw_coords,params.uvw_coords,sizeof(imaging::uvw_coord<uvw_base_type>) * params.row_count,cudaMemcpyHostToDevice,compute_stream));
	cudaSafeCall(cudaMemcpyAsync(gpu_params.baseline_starting_indexes, params.baseline_starting_indexes, sizeof(size_t) * (params.baseline_count+1),cudaMemcpyHostToDevice,compute_stream));
	cudaSafeCall(cudaMemcpyAsync(gpu_params.field_array,params.field_array,sizeof(unsigned int) * params.row_count,cudaMemcpyHostToDevice,compute_stream));
	cudaSafeCall(cudaMemcpyAsync(gpu_params.spw_index_array,params.spw_index_array,sizeof(unsigned int) * params.row_count,
				     cudaMemcpyHostToDevice,compute_stream));
	
	cudaSafeCall(cudaHostUnregister(params.spw_index_array));
	cudaSafeCall(cudaHostUnregister(params.uvw_coords));
	cudaSafeCall(cudaHostUnregister(params.field_array));
	cudaSafeCall(cudaHostUnregister(params.baseline_starting_indexes));
      }
      //invoke computation
      {
	size_t conv_support_size = (params.conv_support*2+1);
	size_t padded_conv_support_size = (conv_support_size+2);
	size_t min_threads_needed = params.baseline_count * conv_support_size * conv_support_size * params.num_facet_centres;
	size_t block_size = NO_THREADS_PER_BLOCK_DIM;
	size_t total_blocks_needed = ceil(min_threads_needed / double(block_size));
	size_t total_blocks_needed_per_dim = total_blocks_needed;
	
	
	dim3 no_threads_per_block(block_size,1,1);
	dim3 no_blocks_per_grid(total_blocks_needed_per_dim,1,1);
	size_t size_of_convolution_function = padded_conv_support_size * params.conv_oversample * sizeof(convolution_base_type); //see algorithms/convolution_policies.h for the reason behind the padding
	typedef imaging::correlation_gridding_policy<imaging::grid_sampling_function> correlation_gridding_policy;
	typedef imaging::baseline_transform_policy<imaging::transform_facet_lefthanded_ra_dec > baseline_transform_policy;
	typedef imaging::phase_transform_policy<imaging::enable_faceting_phase_shift> phase_transform_policy;
	if (params.wplanes > 1)
	  throw std::runtime_error("GPU gridder currently doesn't support w projection options");
	else
	  imaging::templated_gridder<correlation_gridding_policy,baseline_transform_policy,phase_transform_policy><<<no_blocks_per_grid,no_threads_per_block,size_of_convolution_function,compute_stream>>>(gpu_params);
      }
      //swap buffers device -> host when gridded last chunk
      copy_back_sampling_function_if_last_stamp(params,gpu_params);
      gridding_walltime->stop();
    }
}
