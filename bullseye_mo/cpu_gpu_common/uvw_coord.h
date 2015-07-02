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
