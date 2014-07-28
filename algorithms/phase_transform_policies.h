#pragma once

#include <cmath>
#include <complex>
#include <stdexcept>
#include <cstdio>
#include <casa/Quanta/Quantum.h>
#include "uvw_coord.h"

namespace imaging {
	class transform_disable_phase_rotation {};
	class transform_enable_phase_rotation_righthanded_ra_dec {};
	/**
	 Phase transform policies define the behaviour of the gridder; enabling or disabling lm / uv faceting
	 This can be used to effectively decrease computation time when faceting is not needed
	 */
	template <typename visibility_base_type, typename uvw_base_type, typename transform_type>
	class phase_transform_policy {
	public:
		phase_transform_policy() { throw std::runtime_error("Undefined behaviour"); }
		inline void transform (std::complex<visibility_base_type> & __restrict__ visibility,
				       const uvw_coord<uvw_base_type> & __restrict__ uvw) const __restrict__ { 
			throw std::runtime_error("Undefined behaviour"); 
		}
	};
	/**
	 Default policy for disabling faceting.
	 */
	template <typename visibility_base_type, typename uvw_base_type>
	class phase_transform_policy <visibility_base_type, uvw_base_type, transform_disable_phase_rotation> {
	public:
		phase_transform_policy() {}
		inline void transform (std::complex<visibility_base_type> & __restrict__ visibility,
				       const uvw_coord<uvw_base_type> & __restrict__ uvw) const __restrict__ {
			/*Leave unimplemented: serves as the default when no phase shift is required*/
		}
	};
	/**
	 Enable default lm / uv faceting phase shift operations as discussed in Cornwell & Perley (1992) and Synthesis Imaging II (1999)
	 */
	template <typename visibility_base_type, typename uvw_base_type>
	class phase_transform_policy <visibility_base_type, uvw_base_type, transform_enable_phase_rotation_righthanded_ra_dec> {
		uvw_base_type _d_l, _d_m, _d_n;
	public:
		/**
			Convert ra,dec to l,m,n based on Synthesis Imaging II, Pg. 388
			The phase term (as documented in Perley & Cornwell (1992)) calculation requires the delta l,m,n coordinates. 
			Through simplification l0,m0,n0 = (0,0,1) (assume dec == dec0 and ra == ra0, and the simplification follows)
			l,m,n is then calculated using the new and original phase centres as per the relation on Pg. 388
			To limit floating point error the relative phase shift is calulated in arcseconds before converting to radians.
			PRECONDITIONS: the arguements to this method should be measured as arcseconds
		*/
		phase_transform_policy(casa::Quantity old_ra, casa::Quantity old_dec,
				       casa::Quantity new_ra, casa::Quantity new_dec):
				_d_l(0-(cos((new_dec-old_dec).getValue("rad"))*sin((new_ra - old_ra).getValue("rad")))),
				_d_m(0-(sin((new_dec-old_dec).getValue("rad")))),
				_d_n(1-(cos((new_dec-old_dec).getValue("rad"))*cos((new_ra-old_ra).getValue("rad")))){}
		/**
			Performs the transform on the visibility
			Requires uvw coordinates to be measured according to frequency
		*/
		inline void transform (std::complex<visibility_base_type> & __restrict__ visibility,
				       const uvw_coord<uvw_base_type> & __restrict__ uvw) const __restrict__ {
			visibility_base_type x = 2 * M_PI * (uvw._u * _d_l + uvw._v * _d_m + uvw._w * _d_n); //as in Perley & Cornwell (1992)
			std::complex<visibility_base_type> phase_shift_term(cos(x),sin(x)); //by Euler's identity
			visibility *= phase_shift_term;
		}
	};
}
