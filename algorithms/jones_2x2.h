#pragma once

#include <complex>

namespace imaging {
  template <typename visibility_base_type>
  class jones_2x2 {
  public:
    std::complex<visibility_base_type> _polarizations[4];
    static inline void do_hermitian_transpose(jones_2x2<visibility_base_type> & __restrict__ mat){
	mat._polarizations[0].imag(-mat._polarizations[0].imag());
	std::complex<visibility_base_type> swp = std::complex<visibility_base_type>::conj(mat._polarizations[1]);
	mat._polarizations[1] = std::complex<visibility_base_type>::conj(mat._polarizations[2]);
	mat._polarizations[2] = swp;
	mat._polarizations[3].imag(-mat._polarizations[3].imag());
    }
    static inline void do_invert(jones_2x2<visibility_base_type> & __restrict__ mat){
      /*
       if A is a 2x2 matrix, and the determinant is non-zero then:
	A^-1 = det(A)^-1 * |d -b|
			   |-c a|
      */
      std::complex<visibility_base_type> detA = mat._polarizations[0]*mat._polarizations[3] - mat._polarizations[1]*mat._polarizations[2];
      mat._polarizations[1] *= -1;
      mat._polarizations[2] *= -1;
      std::complex<visibility_base_type> swp = std::complex<visibility_base_type>::conj(mat._polarizations[0]);
      mat._polarizations[0] = mat._polarizations[3];
      mat._polarizations[3] = swp;
    }
    static inline void inner_product(const jones_2x2<visibility_base_type> & __restrict__ A,const jones_2x2<visibility_base_type> & __restrict__ B, jones_2x2<visibility_base_type> & __restrict__ out){
      out._polarizations[0] = A._polarizations[0]*B._polarizations[0] + A._polarizations[1]*B._polarizations[2];
      out._polarizations[1] = A._polarizations[0]*B._polarizations[1] + A._polarizations[1]*B._polarizations[3];
      out._polarizations[2] = A._polarizations[2]*B._polarizations[0] + A._polarizations[3]*B._polarizations[2];
      out._polarizations[3] = A._polarizations[2]*B._polarizations[1] + A._polarizations[3]*B._polarizations[3];
    }
  };
}