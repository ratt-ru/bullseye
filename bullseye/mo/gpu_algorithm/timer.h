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
#include <cuda.h>
#include <cuda_runtime.h>
#include "cu_common.h"
namespace utils {
  class timer {
  private:
    cudaEvent_t _start;
    cudaEvent_t _end;
    double _accum;
    cudaStream_t _stream;
    bool _should_record_on_next_start;
  public:
    timer(cudaStream_t stream) : _accum(0), _stream(stream),_should_record_on_next_start(false) {
      cudaSafeCall(cudaEventCreate(&_start));
      cudaSafeCall(cudaEventCreate(&_end));
    }
    virtual ~timer(){
      cudaSafeCall(cudaEventDestroy(_start));
      cudaSafeCall(cudaEventDestroy(_end));
    }
    /**
     * This starts the device timer.
     * The call blocks until previous timing with this timer has been completed
     */
    void start() {
      if (_should_record_on_next_start){
	cudaSafeCall(cudaEventSynchronize(_end)); //ensure it is safe to start (this is effectively a barrier on the specified stream)
	float elapsed = 0;
	cudaSafeCall(cudaEventElapsedTime(&elapsed,_start,_end));
	this->_accum += elapsed/1000.0f;
	_should_record_on_next_start = false;
      }
      cudaSafeCall(cudaEventRecord(_start,_stream));
    }
    /**
     * This stops the device timer
     * The call is placed on the stream specified in the constructor and will execute after all previous asynchronous calls have completed
     */
    void stop() {
      cudaSafeCall(cudaEventRecord(_end,_stream));
      _should_record_on_next_start = true;
    }
    double duration() {
      if (_should_record_on_next_start){
	float elapsed = 0;
	cudaSafeCall(cudaEventSynchronize(_end)); //ensure it is safe to start (this is effectively a barrier on the specified stream)
	cudaSafeCall(cudaEventElapsedTime(&elapsed,_start,_end));
	this->_accum += elapsed/1000.0f;
      }
      return this->_accum;
    }
    void reset() {
      this->_accum = 0;
    }
  };
}