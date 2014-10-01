#include "wrapper.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>
extern "C" {
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
    }
    void releaseLibrary() {
    }
    void weight_uniformly(gridding_parameters & params){
      throw std::runtime_error("Backend Unimplemented exception: weight_uniformly");
    }
    void finalize(gridding_parameters & params){
      throw std::runtime_error("Backend Unimplemented exception: finalize");
    }
    void finalize_psf(gridding_parameters & params){
      throw std::runtime_error("Backend Unimplemented exception: finalize");
    }
    void grid_single_pol(gridding_parameters & params){
      throw std::runtime_error("Backend Unimplemented exception: grid_single_pol");
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