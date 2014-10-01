#pragma once
#include <thread>
#include <future>

#include "timer.h"
#include "gridding_parameters.h"
extern "C" {
    utils::timer gridding_timer;
    utils::timer inversion_timer;
    double get_gridding_walltime() {
      return gridding_timer.duration();
    }
    double get_inversion_walltime() {
      return inversion_timer.duration();
    }
    std::future<void> gridding_future;
    void gridding_barrier() {
        if (gridding_future.valid())
            gridding_future.get(); //Block until result becomes available
    }
    void initLibrary(gridding_parameters & params);
    void releaseLibrary();
    void weight_uniformly(gridding_parameters & params);
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