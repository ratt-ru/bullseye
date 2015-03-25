#pragma once
#include <chrono>
#include <ctime>

namespace utils {
  class timer {
  private:
    std::chrono::time_point<std::chrono::high_resolution_clock> _start;
    double _accum;
  public:
    timer() : _accum(0) {}
    void start() {
      this->_start = std::chrono::high_resolution_clock::now();
    }
    void stop() {
      auto end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> elapsed_seconds = end - this->_start;
      this->_accum += elapsed_seconds.count();
    }
    double duration() {
      return this->_accum;
    }
    void reset() {
      this->_accum = 0;
    }
  };
}