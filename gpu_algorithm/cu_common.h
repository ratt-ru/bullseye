#pragma once
#include <stdint.h>
#include <cufft.h>
/****************************************************************************************************************
 CUDA error handling macros
****************************************************************************************************************/
#define cudaSafeCall(value) {                                                                                   \
        cudaError_t _m_cudaStat = value;                                                                                \
        if (_m_cudaStat != cudaSuccess) {                                                                               \
                fprintf(stderr, "Error %s at line %d in file %s\n",                                     \
                                cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);           \
                exit(1);                                                                                                                        \
        } }

inline void __cufftSafeCall( uint32_t err, const char *file, const int line ){
        if ( CUFFT_SUCCESS != err ){
                fprintf( stderr, "cufftSafeCall() failed at %s:%i\n", file, line);
                exit( -1 );
        }
        return;
}

template <typename T> struct basic_complex { T _real,_imag; };
const float ARCSEC_TO_RAD = M_PI/(180.0*3600.0);