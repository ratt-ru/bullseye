#include <vector>
#include <algorithm>
#include <numeric>
#include <cstring>

#include "gpu_wrapper.h"
#include "dft.h"
#include "templated_gridder.h"

#include "timer.h"
#include "correlation_gridding_traits.h"
#include "correlation_gridding_policies.h"
#include "baseline_transform_policies.h"
#include "phase_transform_policies.h"
#include "jones_2x2.h"

#define NO_THREADS_PER_BLOCK_DIM 256

extern "C" {
    utils::timer * gridding_walltime;
    cudaStream_t compute_stream;
    gridding_parameters gpu_params;
    bool initialized = false;
    double get_gridding_walltime(){
      return gridding_walltime->duration();
    }
    void gridding_barrier(){
      cudaSafeCall(cudaStreamSynchronize(compute_stream));
    }
    void initLibrary(gridding_parameters & params) {
	if (initialized) return;
	initialized = true;
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
            printf("---------------------------------------Backend: GPU GRIDDING LIBRARY---------------------------------------\n");
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
	size_t size_of_convolution_function = (params.conv_support * 2 + 1 + 2) * params.conv_oversample; //see algorithms/convolution_policies.h for the reason behind the padding
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
      cudaSafeCall(cudaStreamDestroy(compute_stream));
      delete gridding_walltime;
      cudaDeviceReset(); //leave the device in a safe state
    }
    void weight_uniformly(gridding_parameters & params){
      throw std::runtime_error("Backend Unimplemented exception: weight_uniformly");
    }
    void copy_back_grid_if_last_stamp(gridding_parameters & params,const gridding_parameters & gpu_params){
      if (params.is_final_data_chunk){
	gridding_barrier();
	cudaSafeCall(cudaMemcpy(params.output_buffer,gpu_params.output_buffer,sizeof(std::complex<grid_base_type>) * params.nx * params.ny * 
				params.number_of_polarization_terms_being_gridded * params.cube_channel_dim_size * params.num_facet_centres,
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
	typedef imaging::baseline_transform_policy<imaging::transform_facet_lefthanded_ra_dec > baseline_transform_policy;
	typedef imaging::phase_transform_policy<imaging::enable_faceting_phase_shift> phase_transform_policy;
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
	  typedef imaging::baseline_transform_policy<imaging::transform_facet_lefthanded_ra_dec > baseline_transform_policy;
	  typedef imaging::phase_transform_policy<imaging::enable_faceting_phase_shift> phase_transform_policy;
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
	  typedef imaging::baseline_transform_policy<imaging::transform_facet_lefthanded_ra_dec > baseline_transform_policy;
	  typedef imaging::phase_transform_policy<imaging::enable_faceting_phase_shift> phase_transform_policy;
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
	  typedef imaging::baseline_transform_policy<imaging::transform_facet_lefthanded_ra_dec > baseline_transform_policy;
	  typedef imaging::phase_transform_policy<imaging::enable_faceting_phase_shift> phase_transform_policy;
	  imaging::templated_gridder<correlation_gridding_policy,baseline_transform_policy,phase_transform_policy><<<no_blocks_per_grid,no_threads_per_block,size_of_convolution_function,compute_stream>>>(gpu_params);
	}
	//swap buffers device -> host when gridded last chunk
	copy_back_grid_if_last_stamp(params,gpu_params);    
      }
      gridding_walltime->stop();
    }
    void grid_sampling_function(gridding_parameters & params){
      gridding_walltime->start();
      printf("Gridding sampling function on the GPU...\n");    
      //copy everything that changed to the gpu
      {
	gpu_params.row_count = params.row_count;
	gpu_params.no_timestamps_read = params.no_timestamps_read;
	gpu_params.is_final_data_chunk = params.is_final_data_chunk;
	//pack and cpy only the necessary visibilities (it doesn't matter if we mod the array here it is not being used again afterwards
	size_t ubound = params.row_count*params.channel_count;
	cudaSafeCall(cudaHostRegister(params.spw_index_array,sizeof(unsigned int) * params.row_count,0));
	cudaSafeCall(cudaHostRegister(params.uvw_coords,sizeof(imaging::uvw_coord<uvw_base_type>) * params.row_count,0));
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
	    params.flags[compact_index] = params.flags[strided_index];
	}
	cudaSafeCall(cudaMemcpyAsync(gpu_params.flags,params.flags,sizeof(bool) * params.row_count * params.channel_count,
				     cudaMemcpyHostToDevice,compute_stream));
	
	cudaSafeCall(cudaHostUnregister(params.spw_index_array));
	cudaSafeCall(cudaHostUnregister(params.uvw_coords));
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
	typedef imaging::correlation_gridding_policy<imaging::grid_sampling_function> correlation_gridding_policy;
	typedef imaging::baseline_transform_policy<imaging::transform_disable_facet_rotation > baseline_transform_policy;
	typedef imaging::phase_transform_policy<imaging::disable_faceting_phase_shift> phase_transform_policy;
	imaging::templated_gridder<correlation_gridding_policy,baseline_transform_policy,phase_transform_policy><<<no_blocks_per_grid,no_threads_per_block,size_of_convolution_function,compute_stream>>>(gpu_params);
      }
      //swap buffers device -> host when gridded last chunk
      copy_back_sampling_function_if_last_stamp(params,gpu_params);
      gridding_walltime->stop();
    }
    void facet_sampling_function(gridding_parameters & params){
      gridding_walltime->start();
      printf("Faceting sampling function on the GPU...\n");    
      //copy everything that changed to the gpu
      {
	gpu_params.row_count = params.row_count;
	gpu_params.no_timestamps_read = params.no_timestamps_read;
	gpu_params.is_final_data_chunk = params.is_final_data_chunk;
	//pack and cpy only the necessary visibilities (it doesn't matter if we mod the array here it is not being used again afterwards
	size_t ubound = params.row_count*params.channel_count;
	cudaSafeCall(cudaHostRegister(params.spw_index_array,sizeof(unsigned int) * params.row_count,0));
	cudaSafeCall(cudaHostRegister(params.uvw_coords,sizeof(imaging::uvw_coord<uvw_base_type>) * params.row_count,0));
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
	    params.flags[compact_index] = params.flags[strided_index];
	}
	cudaSafeCall(cudaMemcpyAsync(gpu_params.visibility_weights,params.visibility_weights,sizeof(visibility_weights_base_type) * params.row_count * params.channel_count,
				     cudaMemcpyHostToDevice,compute_stream));
	cudaSafeCall(cudaMemcpyAsync(gpu_params.flags,params.flags,sizeof(bool) * params.row_count * params.channel_count,
				     cudaMemcpyHostToDevice,compute_stream));
	
	cudaSafeCall(cudaHostUnregister(params.spw_index_array));
	cudaSafeCall(cudaHostUnregister(params.uvw_coords));
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
	typedef imaging::correlation_gridding_policy<imaging::grid_sampling_function> correlation_gridding_policy;
	typedef imaging::baseline_transform_policy<imaging::transform_facet_lefthanded_ra_dec > baseline_transform_policy;
	typedef imaging::phase_transform_policy<imaging::enable_faceting_phase_shift> phase_transform_policy;
	imaging::templated_gridder<correlation_gridding_policy,baseline_transform_policy,phase_transform_policy><<<no_blocks_per_grid,no_threads_per_block,size_of_convolution_function,compute_stream>>>(gpu_params);
      }
      //swap buffers device -> host when gridded last chunk
      copy_back_sampling_function_if_last_stamp(params,gpu_params);
      gridding_walltime->stop();
    }
}
