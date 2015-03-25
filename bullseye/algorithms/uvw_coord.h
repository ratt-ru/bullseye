#pragma once
namespace imaging{
	template<typename T>
	struct uvw_coord{
		T _u,_v,_w;
		uvw_coord(T u = 0, T v = 0, T w = 0):_u(u),_v(v),_w(w){}
	};
	/**
	 * Scalar multiply assign operator
	 */
	template<typename T>
	uvw_coord<T> & __restrict__ operator*= (uvw_coord<T> & __restrict__ lhs, T scalar)  {
	  lhs._u *= scalar;
	  lhs._v *= scalar;
	  lhs._w *= scalar;
	  return lhs;
	}
	/**
	 * Vector scaling operator. Performs an elementwise scaling operation on the uvw coordinate
	 */
	template<typename T>
	uvw_coord<T> & __restrict__ operator*= (uvw_coord<T> & __restrict__ lhs, const uvw_coord<T> & __restrict__ rhs)  {
	  lhs._u *= rhs._u;
	  lhs._v *= rhs._v;
	  lhs._w *= rhs._w;
	  return lhs;
	}
	/**
	 * Negattion operator
	 */
	template<typename T>
	uvw_coord<T> operator- (const uvw_coord<T> & __restrict__ rhs){
	  return uvw_coord<T>(rhs._u*-1,rhs._v*-1,rhs._w*-1);
	}
}
