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
#include "uvw_coord.h"
#include "cu_common.h"
#include "baseline_transform_traits.h"

namespace imaging {
  template <typename T> 
  class baseline_transform_policy {
  public:
  typedef baseline_transform<T> baseline_transform_type;
    __device__ __host__ static void compute_transformation_matrix(uvw_base_type old_phase_centre_ra, uvw_base_type old_phase_centre_dec,
							   uvw_base_type new_phase_centre_ra, uvw_base_type new_phase_centre_dec,
							   baseline_transform_type & result);
    __device__ __host__ static void apply_transformation(uvw_coord<uvw_base_type> & uvw, baseline_transform_type & transformation);
  };
  template<>
  class baseline_transform_policy<transform_disable_facet_rotation> {
  public:
    typedef baseline_transform<transform_disable_facet_rotation> baseline_transform_type;
    __device__ __host__ static void compute_transformation_matrix(uvw_base_type old_phase_centre_ra, uvw_base_type old_phase_centre_dec,
							   uvw_base_type new_phase_centre_ra, uvw_base_type new_phase_centre_dec,
							   baseline_transform_type & result){
      //Leave unimplemented: this should be optimized away
    }
    __device__ __host__ static void apply_transformation(uvw_coord<uvw_base_type> & uvw, baseline_transform_type & transformation){
      //Leave unimplemented: this should be optimized away
    }
  };
  template <> 
  class baseline_transform_policy<transform_facet_lefthanded_ra_dec> {
  public:
    typedef baseline_transform<transform_facet_lefthanded_ra_dec> baseline_transform_type;
    __device__ __host__ static void compute_transformation_matrix(uvw_base_type old_phase_centre_ra, uvw_base_type old_phase_centre_dec,
							   uvw_base_type new_phase_centre_ra, uvw_base_type new_phase_centre_dec,
							   baseline_transform_type & result){
      //see baseline_transform_policy.h in cpu algorithm directory for more details
      //this transformation will let the facet be tangent to the celestial sphere at the new delay centre
      uvw_base_type d_ra = (new_phase_centre_ra - old_phase_centre_ra) * ARCSEC_TO_RAD,
		    n_dec = new_phase_centre_dec * ARCSEC_TO_RAD,
		    o_dec = old_phase_centre_dec * ARCSEC_TO_RAD,
                    c_d_ra = cos(d_ra),
                    s_d_ra = sin(d_ra),
                    c_new_dec = cos(n_dec),
                    c_old_dec = cos(o_dec),
                    s_new_dec = sin(n_dec),
                    s_old_dec = sin(o_dec);
      result.mat_11 = c_d_ra;
      result.mat_12 = s_old_dec*s_d_ra;
      result.mat_13 = -c_old_dec*s_d_ra;
      result.mat_21 = -s_new_dec*s_d_ra;
      result.mat_22 = s_new_dec*s_old_dec*c_d_ra+c_new_dec*c_old_dec;
      result.mat_23 = -c_old_dec*s_new_dec*c_d_ra+c_new_dec*s_old_dec;
      result.mat_31 = c_new_dec*s_d_ra;
      result.mat_32 = -c_new_dec*s_old_dec*c_d_ra+s_new_dec*c_old_dec;
      result.mat_33 = c_new_dec*c_old_dec*c_d_ra+s_new_dec*s_old_dec;
    }
    __device__ __host__ static void apply_transformation(uvw_coord<uvw_base_type> & uvw, const baseline_transform_type & transformation){
      uvw_coord<uvw_base_type> old = uvw;
      uvw._u = transformation.mat_11*old._u + transformation.mat_12*old._v + transformation.mat_13*old._w;
      uvw._v = transformation.mat_21*old._u + transformation.mat_22*old._v + transformation.mat_23*old._w;
      uvw._w = transformation.mat_31*old._u + transformation.mat_32*old._v + transformation.mat_33*old._w;
    }
  };
  template <> 
  class baseline_transform_policy<transform_planar_approx_with_w> {
  public:
    typedef baseline_transform<transform_planar_approx_with_w> baseline_transform_type;
    __device__ __host__ static void compute_transformation_matrix(uvw_base_type old_phase_centre_ra, uvw_base_type old_phase_centre_dec,
							   uvw_base_type new_phase_centre_ra, uvw_base_type new_phase_centre_dec,
							   baseline_transform_type & result){
      //Implements the coordinate uv transform associated with taking a planar approximation to w(n-1) as described in Kogan & Greisen's AIPS Memo 113
      //this is essentially equivalent to rotating the facet to be tangent to the celestial sphere as Perley suggested, but it takes w into account in
      //a linear approximation to the phase error near the facet centre. Of course this 2D taylor expansion of the first order is only valid over a small
      //field of view, but that true of normal tilted faceting as well. Only a convolution can get rid of the (n-1) factor in the ME.
      uvw_base_type d_ra = (new_phase_centre_ra - old_phase_centre_ra) * ARCSEC_TO_RAD,
		    n_dec = new_phase_centre_dec * ARCSEC_TO_RAD,
		    o_dec = old_phase_centre_dec * ARCSEC_TO_RAD,
                    c_d_ra = cos(d_ra),
                    s_d_ra = sin(d_ra),
                    c_new_dec = cos(n_dec),
                    c_old_dec = cos(o_dec),
                    s_new_dec = sin(n_dec),
                    s_old_dec = sin(o_dec),
                    //remember l and m are mearly direction cosines. We relate them to the phase centre using Rick's relations derived from spherical trig (Synthesis Imaging II ch 19, pg 388)
                    li0 = c_new_dec * s_d_ra,
		    mi0 = s_new_dec * c_old_dec - c_new_dec * s_old_dec * c_d_ra,
		    ni0 = s_new_dec * s_old_dec +  c_new_dec * c_old_dec * c_d_ra;
		    // this wiil be muiltiplied by w and subtracted from u and v respectively:
		    result.u_term = li0 / ni0;
		    result.v_term = mi0 / ni0;
    }
    __device__ __host__ static void apply_transformation(uvw_coord<uvw_base_type> & uvw, const baseline_transform_type & transformation){
      uvw._u = uvw._u - uvw._w * transformation.u_term;
      uvw._v = uvw._v - uvw._w * transformation.v_term;
    }
  };
}