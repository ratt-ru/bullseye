#pragma once

#include <cmath>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <stdexcept>
#include <casa/Quanta/Quantum.h>
#include <algorithm>
#include "uvw_coord.h"

namespace imaging {
	class transform_facet_lefthanded_ra_dec {};
	class transform_disable_facet_rotation {};	
	template <typename uvw_base_type,typename coordinate_framework>
	class baseline_transform_policy {
	public:
		baseline_transform_policy() { throw std::runtime_error("Undefined behaviour"); }
		inline uvw_coord<uvw_base_type> transform(uvw_coord<uvw_base_type> baseline) const __restrict__ {
			throw std::runtime_error("Undefined behaviour");
		}
	};
	template <typename uvw_base_type>
	class baseline_transform_policy<uvw_base_type, transform_disable_facet_rotation> {
	public:
		baseline_transform_policy() {}
		inline void transform(uvw_coord<uvw_base_type> & __restrict__ baseline) const __restrict__{
			/*leave unimplemented: serves as "transform" for the no-faceting case*/
		}
	};
	
	template <typename uvw_base_type>
	class baseline_transform_policy <uvw_base_type, transform_facet_lefthanded_ra_dec>{
	private:
		uvw_base_type _baseline_transform_matrix[9];
	public:
		baseline_transform_policy(uvw_base_type facet_original_rotation, uvw_base_type facet_new_rotation, 
                                  	  casa::Quantity old_phase_centre_ra, casa::Quantity old_phase_centre_dec, 
				   	  casa::Quantity new_phase_centre_ra, casa::Quantity new_phase_centre_dec)
		{
			using namespace boost::numeric::ublas;
			/*
                                Compute the following 3x3 coordinate transformation matrix:
                                Z_rot(facet_new_rotation) * T(new_phase_centre_ra,new_phase_centre_dec) * \\
                                        transpose(T(old_phase_centre_ra,old_phase_centre_dec)) * transpose(Z_rot(facet_old_rotation))
                                
                                where:
                                        |       cRA     -sRA.sDEC       sRA.cDEC        |
                                T =     |       -sRA    -cRA.sDEC       cRA.cDEC        |
                                        |       0       cDEC            sDEC            |       
                                 
                        */ 
			uvw_base_type d_ra = (new_phase_centre_ra - old_phase_centre_ra).getValue("rad"),
                                      c_d_ra = cos(d_ra),
                                      s_d_ra = sin(d_ra),
                                      c_new_dec = cos(new_phase_centre_dec.getValue("rad")),
                                      c_old_dec = cos(old_phase_centre_dec.getValue("rad")),
                                      s_new_dec = sin(new_phase_centre_dec.getValue("rad")),
                                      s_old_dec = sin(old_phase_centre_dec.getValue("rad")),
                                      c_orig_rotation = cos(facet_original_rotation),
                                      s_orig_rotation = sin(facet_original_rotation),
                                      c_new_rotation = cos(facet_new_rotation),
                                      s_new_rotation = sin(facet_new_rotation);
			matrix<uvw_base_type> TT_transpose (3,3);
			
			//in row,column format, using basic trig identities the following matrix can be computed (this is the same one used in the classic AIPS task "IMAGR"):
			TT_transpose(0,0) = c_d_ra;
			TT_transpose(0,1) = s_old_dec*s_d_ra;
			TT_transpose(0,2) = -c_old_dec*s_d_ra;
			TT_transpose(1,0) = -s_new_dec*s_d_ra;
                        TT_transpose(1,1) = s_new_dec*s_old_dec*c_d_ra+c_new_dec*c_old_dec;
                        TT_transpose(1,2) = -c_old_dec*s_new_dec*c_d_ra+c_new_dec*s_old_dec;
			TT_transpose(2,0) = c_new_dec*s_d_ra;
                        TT_transpose(2,1) = -c_new_dec*s_old_dec*c_d_ra+s_new_dec*c_old_dec;
                        TT_transpose(2,2) = c_new_dec*c_old_dec*c_d_ra+s_new_dec*s_old_dec;
			
			matrix<uvw_base_type> z_rot_transpose = zero_matrix<uvw_base_type>(3,3);
			z_rot_transpose(0,0) = c_orig_rotation;
			z_rot_transpose(0,1) = s_orig_rotation;
			z_rot_transpose(1,0) = -s_orig_rotation;
			z_rot_transpose(1,1) = c_orig_rotation;
			z_rot_transpose(2,2) = 1;
			
			matrix<uvw_base_type> z_rot = zero_matrix<uvw_base_type>(3,3);
                        z_rot(0,0) = c_new_rotation;
                        z_rot(0,1) = -s_new_rotation;
                        z_rot(1,0) = s_new_rotation;
                        z_rot(1,1) = c_new_rotation;
                        z_rot(2,2) = 1;
			
			matrix<uvw_base_type> full_transform = prod(z_rot,matrix<uvw_base_type>(prod(TT_transpose,z_rot_transpose)));
			//save computed matrix for quick recall:
			std::copy(full_transform.data().begin(),full_transform.data().end(),_baseline_transform_matrix);
		}
                inline void transform(uvw_coord<uvw_base_type> & __restrict__ baseline) const __restrict__{
			//unroll the vector transformation for efficiency:
			uvw_coord<uvw_base_type> old = baseline;
			baseline._u = _baseline_transform_matrix[0]*old._u + _baseline_transform_matrix[1]*old._v + _baseline_transform_matrix[2]*old._w;
			baseline._v = _baseline_transform_matrix[0+3]*old._u + _baseline_transform_matrix[1+3]*old._v + _baseline_transform_matrix[2+3]*old._w;
			baseline._w = _baseline_transform_matrix[0+6]*old._u + _baseline_transform_matrix[1+6]*old._v + _baseline_transform_matrix[2+6]*old._w;
		}
        };
}
