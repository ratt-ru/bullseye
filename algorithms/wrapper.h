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