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
extern "C" {
    double get_gridding_walltime();
    double get_inversion_walltime();
    void gridding_barrier();
    void initLibrary(gridding_parameters & params);
    void releaseLibrary();
    void weight_uniformly(gridding_parameters & params);
    void normalize(gridding_parameters & params);
    void repack_input_data(gridding_parameters & params);
    void finalize(gridding_parameters & params);
    void finalize_psf(gridding_parameters & params);
    void grid_single_pol(gridding_parameters & params);
    void facet_single_pol(gridding_parameters & params);
    void grid_duel_pol(gridding_parameters & params);
    void facet_duel_pol(gridding_parameters & params);
    void grid_4_cor(gridding_parameters & params);
    void facet_4_cor(gridding_parameters & params);
    void facet_4_cor_corrections(gridding_parameters & params);
    void grid_sampling_function(gridding_parameters & params);
    void facet_sampling_function(gridding_parameters & params);
}