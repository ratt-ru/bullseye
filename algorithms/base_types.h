#pragma once

#ifdef BULLSEYE_SINGLE
  typedef float visibility_base_type;
  typedef float uvw_base_type;
  typedef float reference_wavelengths_base_type;
  typedef float convolution_base_type;
  typedef float visibility_weights_base_type;
  typedef float grid_base_type;
  typedef double normalization_base_type;
  #define SHOULD_DO_32_BIT_FFT
#endif
#ifdef BULLSEYE_DOUBLE
  typedef double visibility_base_type;
  typedef double uvw_base_type;
  typedef double reference_wavelengths_base_type;
  typedef double convolution_base_type;
  typedef double visibility_weights_base_type;
  typedef double grid_base_type;
  typedef double normalization_base_type;
#endif