#pragma once
/**
 * These usefull block indexing schemes I borrowed from
 * http://www.martinpeniak.com/index.php?option=com_content&view=article&catid=17:updates&id=288:cuda-thread-indexing-explained
 * Original author: Martin Peniak
 */
namespace cu_indexing_schemes {
  __device__ size_t getGlobalIdx_1D_1D(const dim3 & gridDim, const dim3 & blockIdx, 
				       const dim3 & blockDim, const dim3 & threadIdx)
  {
    return blockIdx.x *blockDim.x + threadIdx.x;
  }

  __device__ size_t getGlobalIdx_1D_2D(const dim3 & gridDim, const dim3 & blockIdx, 
				       const dim3 & blockDim, const dim3 & threadIdx)
  {
    return blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
  }

  __device__ size_t getGlobalIdx_1D_3D(const dim3 & gridDim, const dim3 & blockIdx, 
				       const dim3 & blockDim, const dim3 & threadIdx)
  {
    return blockIdx.x * blockDim.x * blockDim.y * blockDim.z +
           threadIdx.z * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x;
  }

  __device__ size_t getGlobalIdx_2D_1D(const dim3 & gridDim, const dim3 & blockIdx, 
				       const dim3 & blockDim, const dim3 & threadIdx)
  {
    size_t blockId = blockIdx.y * gridDim.x + blockIdx.x;
    size_t threadId = blockId * blockDim.x + threadIdx.x;
    return threadId;
  }

  __device__ size_t getGlobalIdx_2D_2D(const dim3 & gridDim, const dim3 & blockIdx, 
				       const dim3 & blockDim, const dim3 & threadIdx)
  {
    size_t blockId = blockIdx.x + blockIdx.y * gridDim.x;
    size_t threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
    return threadId;
  }

  __device__ size_t getGlobalIdx_2D_3D(const dim3 & gridDim, const dim3 & blockIdx, 
				       const dim3 & blockDim, const dim3 & threadIdx)
  {
    size_t blockId = blockIdx.x + blockIdx.y * gridDim.x;
    size_t threadId = blockId * (blockDim.x * blockDim.y * blockDim.z) +
                      (threadIdx.z * (blockDim.x * blockDim.y)) +
                      (threadIdx.y * blockDim.x) +
                      threadIdx.x;
    return threadId;
  }

  __device__ size_t getGlobalIdx_3D_1D(const dim3 & gridDim, const dim3 & blockIdx, 
				       const dim3 & blockDim, const dim3 & threadIdx)
  {
    size_t blockId = blockIdx.x +
                     blockIdx.y * gridDim.x +
                     gridDim.x * gridDim.y * blockIdx.z;
    size_t threadId = blockId * blockDim.x + threadIdx.x;
    return threadId;
  }

  __device__ size_t getGlobalIdx_3D_2D(const dim3 & gridDim, const dim3 & blockIdx, 
				       const dim3 & blockDim, const dim3 & threadIdx)
  {
    size_t blockId = blockIdx.x +
                     blockIdx.y * gridDim.x +
                     gridDim.x * gridDim.y * blockIdx.z;
    size_t threadId = blockId * (blockDim.x * blockDim.y) +
                      (threadIdx.y * blockDim.x) +
                      threadIdx.x;
    return threadId;
  }

  __device__ size_t getGlobalIdx_3D_3D(const dim3 & gridDim, const dim3 & blockIdx, 
				       const dim3 & blockDim, const dim3 & threadIdx)
  {
    size_t blockId = blockIdx.x +
                     blockIdx.y * gridDim.x +
                     gridDim.x * gridDim.y * blockIdx.z;
    size_t threadId = blockId * (blockDim.x * blockDim.y * blockDim.z) +
                      (threadIdx.z * (blockDim.x * blockDim.y)) +
                      (threadIdx.y * blockDim.x) +
                      threadIdx.x;
    return threadId;
  }
}