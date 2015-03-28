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
#include "gridding_parameters.h"
#include "uvw_coord.h"
#include "cu_common.h"

namespace imaging {
  class transform_facet_lefthanded_ra_dec {};
  class transform_disable_facet_rotation {};
  struct baseline_rotation_mat {
    uvw_base_type mat_11, mat_12, mat_13,
		  mat_21, mat_22, mat_23,
		  mat_31, mat_32, mat_33;
  };
  template <typename T> 
  class baseline_transform_policy {
  public:
    __device__ __host__ static void compute_transformation_matrix(uvw_base_type old_phase_centre_ra, uvw_base_type old_phase_centre_dec,
							   uvw_base_type new_phase_centre_ra, uvw_base_type new_phase_centre_dec,
							   baseline_rotation_mat & result);
    __device__ __host__ static void apply_transformation(uvw_coord<uvw_base_type> & uvw, baseline_rotation_mat & transformation_matrix);
  };
  template<>
  class baseline_transform_policy<transform_disable_facet_rotation> {
  public:
    __device__ __host__ static void compute_transformation_matrix(uvw_base_type old_phase_centre_ra, uvw_base_type old_phase_centre_dec,
							   uvw_base_type new_phase_centre_ra, uvw_base_type new_phase_centre_dec,
							   baseline_rotation_mat & result){
      //Leave unimplemented: this should be optimized away
    }
    __device__ __host__ static void apply_transformation(uvw_coord<uvw_base_type> & uvw, baseline_rotation_mat & transformation_matrix){
      //Leave unimplemented: this should be optimized away
    }
  };
  template <> 
  class baseline_transform_policy<transform_facet_lefthanded_ra_dec> {
  public:
    __device__ __host__ static void compute_transformation_matrix(uvw_base_type old_phase_centre_ra, uvw_base_type old_phase_centre_dec,
							   uvw_base_type new_phase_centre_ra, uvw_base_type new_phase_centre_dec,
							   baseline_rotation_mat & result){
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
    __device__ __host__ static void apply_transformation(uvw_coord<uvw_base_type> & uvw, const baseline_rotation_mat & transformation_matrix){
      uvw_coord<uvw_base_type> old = uvw;
      uvw._u = transformation_matrix.mat_11*old._u + transformation_matrix.mat_12*old._v + transformation_matrix.mat_13*old._w;
      uvw._v = transformation_matrix.mat_21*old._u + transformation_matrix.mat_22*old._v + transformation_matrix.mat_23*old._w;
      uvw._w = transformation_matrix.mat_31*old._u + transformation_matrix.mat_32*old._v + transformation_matrix.mat_33*old._w;
    }
  };
}