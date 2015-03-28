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
#include <complex>
#include "jones_2x2.h"
namespace imaging {
  class gridding_single_pol {};
  class gridding_double_pol {};
  class gridding_4_pol {};
  class gridding_4_pol_enable_facet_based_jones_corrections {};
  class gridding_sampling_function {};
  template <typename visibility_base_type,typename weights_base_type,typename T>
  class polarization_gridding_trait {
    //Undefined base class
    class undefined_type {};
  public:
    typedef undefined_type pol_vis_type;
    typedef undefined_type pol_vis_weight_type;
    typedef undefined_type pol_vis_flag_type;
  };
  
  template <typename visibility_base_type,typename weights_base_type>
  class polarization_gridding_trait<visibility_base_type,weights_base_type,gridding_single_pol> {
  public:
    typedef std::complex<visibility_base_type> pol_vis_type;
    typedef weights_base_type pol_vis_weight_type;
    typedef bool pol_vis_flag_type;
  };
  
  template <typename visibility_base_type,typename weights_base_type>
  class polarization_gridding_trait<visibility_base_type,weights_base_type,gridding_sampling_function>: 
    public polarization_gridding_trait<visibility_base_type,weights_base_type,gridding_single_pol>{
  };
  
  template <typename visibility_base_type,typename weights_base_type>
  class polarization_gridding_trait<visibility_base_type,weights_base_type,gridding_double_pol> {
  public:
    typedef struct pol_vis_type {std::complex<visibility_base_type> v[2]; } pol_vis_type;
    typedef struct pol_vis_weight_type { weights_base_type w[2]; } pol_vis_weight_type;
    typedef struct pol_vis_flag_type { bool f[2]; } pol_vis_flag_type;
  };
  
  template <typename visibility_base_type,typename weights_base_type>
  class polarization_gridding_trait<visibility_base_type,weights_base_type,gridding_4_pol> {
  public:
    typedef jones_2x2<visibility_base_type> pol_vis_type;
    typedef struct pol_vis_weight_type { weights_base_type w[4]; } pol_vis_weight_type;
    typedef struct pol_vis_flag_type { bool f[4]; } pol_vis_flag_type;
  };
  
  template <typename visibility_base_type,typename weights_base_type>
  class polarization_gridding_trait<visibility_base_type,weights_base_type,gridding_4_pol_enable_facet_based_jones_corrections> : 
    public polarization_gridding_trait<visibility_base_type,weights_base_type,gridding_4_pol> {
  };
}