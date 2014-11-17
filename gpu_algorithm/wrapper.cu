#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <cstdio>
#include "wrapper.h"
#include "dft.h"
#include "gridder.h"
#include "timer.h"
#include "cu_common.h"
#include "uvw_coord.h"
#include "fft_shift_utils.h"
#define NO_THREADS_PER_BLOCK_DIM 16

extern imaging::uvw_coord< double > uvw;
extern "C" {
    utils::timer * gridding_walltime;
    utils::timer * inversion_walltime;
    cudaStream_t compute_stream;
    
    gridding_parameters gpu_params;
    
    double get_gridding_walltime(){
      return gridding_walltime->duration();
    }
    double get_inversion_walltime(){
      return inversion_walltime->duration();
    }
    void gridding_barrier(){
      cudaSafeCall(cudaStreamSynchronize(compute_stream));
    }
    void initLibrary(gridding_parameters & params) {
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
            printf("-----------------------------------Backend: GPU DFT Library---------------------------------------\n");
            printf("%s, device %d on PCI Bus #%d, clocked at %f GHz\n",properties.name,properties.pciDeviceID,
                   properties.pciBusID,properties.clockRate / 1000000.0);
            printf("Compute capability %d.%d with %f GiB global memory (%f GiB free)\n",properties.major,
                   properties.minor,mem_tot/1024.0/1024.0/1024.0,mem_free/1024.0/1024.0/1024.0);
            printf("%d SMs are available\n",properties.multiProcessorCount);
            printf("--------------------------------------------------------------------------------------------------\n");
        } else 
            throw std::runtime_error("Cannot find suitable GPU device. Giving up");
	cudaSafeCall(cudaStreamCreateWithFlags(&compute_stream,cudaStreamNonBlocking));
	gridding_walltime = new utils::timer(compute_stream);
	inversion_walltime = new utils::timer(compute_stream);
	//alloc memory for all the arrays on the gpu at the beginning of execution...
	gpu_params = params;
	cudaSafeCall(cudaMalloc((void**)&gpu_params.visibilities, sizeof(std::complex<visibility_base_type>) * params.chunk_max_row_count*params.channel_count*params.number_of_polarization_terms_being_gridded));
	cudaSafeCall(cudaMalloc((void**)&gpu_params.uvw_coords, sizeof(imaging::uvw_coord<uvw_base_type>) * params.chunk_max_row_count));
	cudaSafeCall(cudaMalloc((void**)&gpu_params.reference_wavelengths, sizeof(reference_wavelengths_base_type) * params.channel_count));
	cudaSafeCall(cudaMemcpy(gpu_params.reference_wavelengths,params.reference_wavelengths,sizeof(reference_wavelengths_base_type) * params.channel_count,cudaMemcpyHostToDevice));
	cudaSafeCall(cudaMalloc((void**)&gpu_params.enabled_channels, sizeof(bool) * params.channel_count));
	cudaSafeCall(cudaMemcpy(gpu_params.enabled_channels,params.enabled_channels, sizeof(bool) * params.channel_count,cudaMemcpyHostToDevice));
	cudaSafeCall(cudaMalloc((void**)&gpu_params.channel_grid_indicies, sizeof(size_t) * params.channel_count));
	cudaSafeCall(cudaMemcpy(gpu_params.channel_grid_indicies,params.channel_grid_indicies, sizeof(size_t) * params.channel_count,cudaMemcpyHostToDevice));
	cudaSafeCall(cudaMalloc((void**)&gpu_params.spw_index_array, sizeof(unsigned int) * params.chunk_max_row_count));
	cudaSafeCall(cudaMalloc((void**)&gpu_params.flagged_rows, sizeof(bool) * params.chunk_max_row_count));
	cudaSafeCall(cudaMalloc((void**)&gpu_params.visibility_weights, sizeof(visibility_weights_base_type) * params.chunk_max_row_count * params.channel_count * params.number_of_polarization_terms_being_gridded));
	cudaSafeCall(cudaMalloc((void**)&gpu_params.flags, sizeof(bool) * params.chunk_max_row_count * params.channel_count * params.number_of_polarization_terms_being_gridded));
	cudaSafeCall(cudaMalloc((void**)&gpu_params.output_buffer, sizeof(std::complex<grid_base_type>) * params.nx * params.ny * params.number_of_polarization_terms_being_gridded * params.cube_channel_dim_size));
	cudaSafeCall(cudaMemset(gpu_params.output_buffer,0,sizeof(grid_base_type) * params.nx * params.ny * params.number_of_polarization_terms_being_gridded * params.cube_channel_dim_size));
	cudaSafeCall(cudaMalloc((void**)&gpu_params.conv, sizeof(convolution_base_type) * params.conv_support * params.conv_oversample));	
	cudaSafeCall(cudaMemcpy(gpu_params.conv, params.conv, sizeof(convolution_base_type) * params.conv_support * params.conv_oversample,cudaMemcpyHostToDevice));
	
	cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
    }
    void releaseLibrary() {
      cudaDeviceSynchronize();
      cudaSafeCall(cudaFree(gpu_params.output_buffer));
      cudaSafeCall(cudaFree(gpu_params.visibilities));
      cudaSafeCall(cudaFree(gpu_params.uvw_coords));
      cudaSafeCall(cudaFree(gpu_params.reference_wavelengths));
      cudaSafeCall(cudaFree(gpu_params.enabled_channels));
      cudaSafeCall(cudaFree(gpu_params.channel_grid_indicies));
      cudaSafeCall(cudaFree(gpu_params.spw_index_array));
      cudaSafeCall(cudaFree(gpu_params.flagged_rows));
      cudaSafeCall(cudaFree(gpu_params.flags));
      cudaSafeCall(cudaFree(gpu_params.conv));
      cudaSafeCall(cudaStreamDestroy(compute_stream));
      cudaDeviceReset(); //leave the device in a safe state
    }
    void weight_uniformly(gridding_parameters & params){
      throw std::runtime_error("Backend Unimplemented exception: weight_uniformly");
    }
    void finalize(gridding_parameters & params){
      gridding_barrier();
      inversion_walltime->start();
      std::size_t offset = params.nx*params.ny*params.cube_channel_dim_size*params.number_of_polarization_terms_being_gridded;
      /*
       * We'll be storing 32 bit real fits files so ignore all the imaginary components and cast whatever the grid was to float32
       */
      {
	  grid_base_type * __restrict__ grid_ptr_gridtype = (grid_base_type *)params.output_buffer;
	  float * __restrict__ grid_ptr_single = (float *)params.output_buffer;
	  for (std::size_t f = 0; f < params.num_facet_centres; ++f) {
	      std::size_t casting_lbound = offset*f;
	      std::size_t casting_ubound = casting_lbound + params.nx*params.ny*params.cube_channel_dim_size;
	      for (std::size_t i = casting_lbound; i < casting_ubound; ++i){
		  grid_ptr_single[i] = (float)(grid_ptr_gridtype[i]); //extract all the reals
	      }
	  }
      }
      inversion_walltime->stop();
    }
    void finalize_psf(gridding_parameters & params){
      throw std::runtime_error("Backend Unimplemented exception: finalize_psf");
    }
    void grid_single_pol(gridding_parameters & params){
      gridding_barrier();
      gridding_walltime->start();
      printf("Gridding single polarization on the GPU...\n");    
      //copy everything that changed to the gpu
      {
	gpu_params.row_count = params.row_count;
	gpu_params.no_timestamps_read = params.no_timestamps_read;
	cudaSafeCall(cudaMemcpyAsync(gpu_params.uvw_coords,params.uvw_coords,sizeof(imaging::uvw_coord<uvw_base_type>) * params.row_count,cudaMemcpyHostToDevice,compute_stream));
	//pack and cpy only the necessary visibilities (it doesn't matter if we mod the array here it is not being used again afterwards
	size_t ubound = params.row_count*params.channel_count;
	for (std::size_t i = 0; i < ubound; ++i){
	    size_t r = i / params.channel_count;
	    size_t c = i % params.channel_count;
	    size_t compact_index = r*params.channel_count + c;
	    size_t strided_index = (r*params.number_of_polarization_terms + params.polarization_index)*params.channel_count + c;
	    
	    params.visibilities[compact_index] = params.visibilities[strided_index];
	    params.visibility_weights[compact_index] = params.visibility_weights[strided_index];
	    params.flags[compact_index] = params.flags[strided_index];
	}
	cudaSafeCall(cudaMemcpyAsync(gpu_params.visibilities,params.visibilities,sizeof(std::complex<visibility_base_type>) * params.row_count * params.channel_count,
				     cudaMemcpyHostToDevice,compute_stream));
	cudaSafeCall(cudaMemcpyAsync(gpu_params.spw_index_array,params.spw_index_array,sizeof(unsigned int) * params.row_count,
				     cudaMemcpyHostToDevice,compute_stream));
	cudaSafeCall(cudaMemcpyAsync(gpu_params.visibility_weights,params.visibility_weights,sizeof(visibility_weights_base_type) * params.row_count * params.channel_count,
				     cudaMemcpyHostToDevice,compute_stream));
	cudaSafeCall(cudaMemcpyAsync(gpu_params.flagged_rows,params.flagged_rows,sizeof(bool) * params.row_count,
				     cudaMemcpyHostToDevice,compute_stream));
	cudaSafeCall(cudaMemcpyAsync(gpu_params.flags,params.flags,sizeof(bool) * params.row_count * params.channel_count,
				     cudaMemcpyHostToDevice,compute_stream));
      }
      //invoke computation
      {
	dim3 no_blocks_per_grid(ceil(params.nx / (double)NO_THREADS_PER_BLOCK_DIM),
				ceil(params.ny / (double)NO_THREADS_PER_BLOCK_DIM),
				1);
	dim3 no_threads_per_block(NO_THREADS_PER_BLOCK_DIM,NO_THREADS_PER_BLOCK_DIM,1);
	imaging::grid_single<<<no_blocks_per_grid,no_threads_per_block,0,compute_stream>>>(gpu_params,no_blocks_per_grid,no_threads_per_block);
      }
      //swap buffers device -> host when gridded last chunk
      if (params.is_final_data_chunk){
	gridding_barrier();
	cudaSafeCall(cudaMemcpy(params.output_buffer,gpu_params.output_buffer,sizeof(grid_base_type) * params.nx * params.ny,cudaMemcpyDeviceToHost));
      }      
      gridding_walltime->stop();
    }
    void facet_single_pol(gridding_parameters & params){
      throw std::runtime_error("Backend Unimplemented exception: facet_single_pol");
    }
    void grid_duel_pol(gridding_parameters & params){
      throw std::runtime_error("Backend Unimplemented exception: grid_duel_pol");
    }
    void facet_duel_pol(gridding_parameters & params){
      throw std::runtime_error("Backend Unimplemented exception: facet_duel_pol");
    }
    void grid_4_cor(gridding_parameters & params){
      throw std::runtime_error("Backend Unimplemented exception: grid_4_cor");
    }
    void facet_4_cor(gridding_parameters & params){
      throw std::runtime_error("Backend Unimplemented exception: facet_4_cor");
    }
    void facet_4_cor_corrections(gridding_parameters & params){
      throw std::runtime_error("Backend Unimplemented exception: facet_4_cor_corrections");
    }
    void grid_sampling_function(gridding_parameters & params){
      throw std::runtime_error("Backend Unimplemented exception: grid_sampling_function");
    }
    void facet_sampling_function(gridding_parameters & params){
      throw std::runtime_error("Backend Unimplemented exception: facet_sampling_function");
    }
}
