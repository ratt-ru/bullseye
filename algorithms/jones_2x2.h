#pragma once

#include <complex>

namespace imaging {
  template <typename visibility_base_type>
  struct jones_2x2 {
    std::complex<visibility_base_type> correlations[4];    
  };
  
  /**
   * Element-wise conjugates the input matrix and transposes this conjugate matrix
   */
  template <typename visibility_base_type>
  inline void do_hermitian_transpose(jones_2x2<visibility_base_type> & __restrict__ mat) {
    mat.correlations[0].imag(-mat.correlations[0].imag());
    std::complex<visibility_base_type> swp = conj(mat.correlations[1]);
    mat.correlations[1] = conj(mat.correlations[2]);
    mat.correlations[2] = swp;
    mat.correlations[3].imag(-mat.correlations[3].imag());
  }
  /**
   * Calculates the determinant of a 2x2 complex matrix
   */
  template <typename visibility_base_type>
  inline std::complex<visibility_base_type> det(const jones_2x2<visibility_base_type> & __restrict__ mat){
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
    std::complex<visibility_base_type> detInv = std::complex<visibility_base_type>(1,0)/det(mat);
    mat.correlations[1] *= -detInv;
    mat.correlations[2] *= -detInv;
    std::complex<visibility_base_type> swp = mat.correlations[0];
    mat.correlations[0] = mat.correlations[3] * detInv;
    mat.correlations[3] = swp * detInv;
    
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