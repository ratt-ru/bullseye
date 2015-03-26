#pragma once

#ifdef __CUDACC__
#include "cu_basic_complex.h"
#else
#include <complex>
#endif

namespace imaging {
  template <typename visibility_base_type>
  struct jones_2x2 {
    #ifdef __CUDACC__
      basic_complex<visibility_base_type> correlations[4];
    #else
      std::complex<visibility_base_type> correlations[4];
    #endif  
  };
  
  /**
   * Element-wise conjugates the input matrix and transposes this conjugate matrix
   */
  template <typename visibility_base_type>
  inline void do_hermitian_transpose(jones_2x2<visibility_base_type> & __restrict__ mat) {
    mat.correlations[0].imag(-mat.correlations[0].imag());
    #ifdef __CUDACC__
      basic_complex<visibility_base_type> swp = conj(mat.correlations[1]);
    #else
      std::complex<visibility_base_type> swp = conj(mat.correlations[1]);
    #endif
    mat.correlations[1] = conj(mat.correlations[2]);
    mat.correlations[2] = swp;
    mat.correlations[3].imag(-mat.correlations[3].imag());
  }
  /**
   * Calculates the determinant of a 2x2 complex matrix
   */
  template <typename visibility_base_type>
  #ifdef __CUDACC__
  inline basic_complex<visibility_base_type> det(const jones_2x2<visibility_base_type> & __restrict__ mat){
  #else
  inline std::complex<visibility_base_type> det(const jones_2x2<visibility_base_type> & __restrict__ mat){
  #endif
    return (mat.correlations[0]*mat.correlations[3] - mat.correlations[1]*mat.correlations[2]);
  }
  /**
   * Calculates the inverse of a 2x2 complex matrix
   */
  template <typename visibility_base_type>
  inline void invert(jones_2x2<visibility_base_type> & __restrict__ mat) {
    /*
      if A is a 2x2 matrix, and the determinant is non-zero then:
      A^-1 = det(A)^-1 * |d -b|
			 |-c a|
    */
    #ifdef __CUDACC__
      basic_complex<visibility_base_type> detInv = basic_complex<visibility_base_type>(1,0)/det(mat);
    #else
      std::complex<visibility_base_type> detInv = std::complex<visibility_base_type>(1,0)/det(mat);
    #endif
    mat.correlations[1] *= -detInv;
    mat.correlations[2] *= -detInv;
    #ifdef __CUDACC__
      basic_complex<visibility_base_type> swp = mat.correlations[0];
    #else
      std::complex<visibility_base_type> swp = mat.correlations[0];
    #endif
    mat.correlations[0] = mat.correlations[3] * detInv;
    mat.correlations[3] = swp * detInv;
  }
  /**
   * Inverts all matricies in the set
   * Assumes all the matricies are non-singular, will throw an exception if a 
   * singular matrix is encountered.
   */
  template <typename visibility_base_type>
  inline void invert_all(jones_2x2<visibility_base_type> * __restrict__ mat, std::size_t jones_count) {
    #pragma omp parallel for
    for (std::size_t j = 0; j < jones_count; ++j){
      #ifdef __CUDACC__
      if (det(mat[j]) == basic_complex<visibility_base_type>(0,0)){
      #else
      if (det(mat[j]) == std::complex<visibility_base_type>(0,0)){
      #endif
	throw std::runtime_error("JONES MATRIX SET CONTAINS SINGULAR MATRICIES. ABORTING.");
      }
      invert(mat[j]);
    }
  }
  /**
   * Unrolled 2x2 matrix multiplication
   * Assumption A != out and B != out (not safe for inplace matrix multiplication.
   * See inner_product_inplace for alternative
   */
  template <typename visibility_base_type>
  inline void inner_product(const jones_2x2<visibility_base_type> & __restrict__ A,
				   const jones_2x2<visibility_base_type> & __restrict__ B, 
				   jones_2x2<visibility_base_type> & __restrict__ out) {
    //this will NOT work for an inplace operation
    out.correlations[0] = A.correlations[0]*B.correlations[0] + A.correlations[1]*B.correlations[2];
    out.correlations[1] = A.correlations[0]*B.correlations[1] + A.correlations[1]*B.correlations[3];
    out.correlations[2] = A.correlations[2]*B.correlations[0] + A.correlations[3]*B.correlations[2];
    out.correlations[3] = A.correlations[2]*B.correlations[1] + A.correlations[3]*B.correlations[3];
  }
  /**
   * Unrolled 2x2 matrix multiplication
   * It is safe to use this operation for inplace multiplication
   */
  template <typename visibility_base_type>
  inline void inner_product_inplace(const jones_2x2<visibility_base_type> & __restrict__ A,
				    const jones_2x2<visibility_base_type> & __restrict__ B, 
				    jones_2x2<visibility_base_type> & __restrict__ out) {
    //make deep copy to guarentee inplace will work
    jones_2x2<visibility_base_type> A_cpy = A; 
    jones_2x2<visibility_base_type> B_cpy = B;
    out.correlations[0] = A_cpy.correlations[0]*B_cpy.correlations[0] + A_cpy.correlations[1]*B_cpy.correlations[2];
    out.correlations[1] = A_cpy.correlations[0]*B_cpy.correlations[1] + A_cpy.correlations[1]*B_cpy.correlations[3];
    out.correlations[2] = A_cpy.correlations[2]*B_cpy.correlations[0] + A_cpy.correlations[3]*B_cpy.correlations[2];
    out.correlations[3] = A_cpy.correlations[2]*B_cpy.correlations[1] + A_cpy.correlations[3]*B_cpy.correlations[3];
  }
}