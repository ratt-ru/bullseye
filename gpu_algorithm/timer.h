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