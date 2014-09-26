#pragma once
#include "timer.h"
#include "gridding_parameters.h"
#include <thread>
#include <future>

extern "C" {
    utils::timer gridding_timer;
    double get_gridding_walltime() {
      return gridding_timer.duration();
    }
    std::future<void> gridding_future;
    void gridding_barrier() {
        if (gridding_future.valid())
            gridding_future.get(); //Block until result becomes available
    }
    void initLibrary();
    void releaseLibrary();
    void weight_uniformly(gridding_parameters & params);
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