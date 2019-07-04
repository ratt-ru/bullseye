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