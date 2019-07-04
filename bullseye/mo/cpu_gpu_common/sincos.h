#pragma once
#ifdef BULLSEYE_SINGLE
  #ifdef __CUDACC__
    __device__ void custom_sincos(float arg, float * s, float * c){ __sincosf(arg,s,c); }
  #else
    void custom_sincos(float arg, float * s, float * c){ sincosf(arg,s,c); }
  #endif
#elif BULLSEYE_DOUBLE
  #ifdef __CUDACC__
  __device__ void custom_sincos(double arg, double * s, double * c){ sincos(arg,s,c); }
  #else
    void custom_sincos(double arg, double * s, double * c){ sincos(arg,s,c); }
  #endif
#endif