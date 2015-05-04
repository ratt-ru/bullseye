#pragma once
#include "uvw_coord.h"
namespace imaging {
class convolution_AA_1D_precomputed {};

template <typename active_correlation_gridding_policy,typename T>
class convolution_policy {
public:
    inline static void convolve(gridding_parameters & params, uvw_base_type grid_centre_offset_x,
				
                                uvw_base_type grid_centre_offset_y,
                                grid_base_type * __restrict__ facet_output_buffer,
				std::size_t channel_grid_index,
                                std::size_t grid_size_in_floats,
				size_t conv_full_support,
				size_t padded_conv_full_support,
				uvw_coord< uvw_base_type > & uvw,
                                typename active_correlation_gridding_policy::active_trait::vis_type & vis,
                                typename active_correlation_gridding_policy::active_trait::normalization_accumulator_type & normalization_term);
};

template <typename active_correlation_gridding_policy>
class convolution_policy <active_correlation_gridding_policy,convolution_AA_1D_precomputed> {
public:
    inline static void convolve(gridding_parameters & params, uvw_base_type grid_centre_offset_x,
                                uvw_base_type grid_centre_offset_y,
                                grid_base_type * __restrict__ facet_output_buffer,
				std::size_t channel_grid_index,
                                std::size_t grid_size_in_floats,
				size_t conv_full_support,
				size_t padded_conv_full_support,
				uvw_coord< uvw_base_type > & uvw,
                                typename active_correlation_gridding_policy::active_trait::vis_type & vis,
                                typename active_correlation_gridding_policy::active_trait::normalization_accumulator_type & normalization_term) {
        //account for interpolation error (we select the closest sample from the oversampled convolution filter)
        uvw_base_type translated_grid_u = uvw._u + grid_centre_offset_x;
        uvw_base_type translated_grid_v = uvw._v + grid_centre_offset_y;
        std::size_t  disc_grid_u = std::lrint(translated_grid_u);
        std::size_t  disc_grid_v = std::lrint(translated_grid_v);
        //to reduce the interpolation error we need to take the offset from the grid centre into account when choosing a convolution weight
        uvw_base_type frac_u = -translated_grid_u + (uvw_base_type)disc_grid_u;
        uvw_base_type frac_v = -translated_grid_v + (uvw_base_type)disc_grid_v;
        //Don't you dare go over the boundary
        if (disc_grid_v + padded_conv_full_support  >= params.ny || disc_grid_u + padded_conv_full_support >= params.nx ||
                disc_grid_v >= params.ny || disc_grid_u >= params.nx) return;

        std::size_t conv_v = (frac_v + 1) * params.conv_oversample;
        std::size_t  convolved_grid_v = (disc_grid_v + 1)*params.nx;
        for (std::size_t  sup_v = 1; sup_v <= conv_full_support; ++sup_v) { //remember we have a +/- frac at both ends of the filter
            convolution_base_type conv_v_weight = params.conv[conv_v];
            std::size_t conv_u = (frac_u + 1) * params.conv_oversample;
            for (std::size_t sup_u = 1; sup_u <= conv_full_support; ++sup_u) { //remember we have a +/- frac at both ends of the filter
                std::size_t convolved_grid_u = disc_grid_u + sup_u;
                convolution_base_type conv_u_weight = params.conv[conv_u];
                std::size_t grid_flat_index = convolved_grid_v + convolved_grid_u;

                convolution_base_type conv_weight = conv_u_weight * conv_v_weight;
                typename active_correlation_gridding_policy::active_trait::vis_type convolved_vis = vis * conv_weight;
                active_correlation_gridding_policy::grid_visibility(facet_output_buffer,
                        grid_size_in_floats,
                        params.nx,
                        channel_grid_index,
                        params.number_of_polarization_terms_being_gridded,
                        disc_grid_u + sup_u,
                        disc_grid_v + sup_v,
                        convolved_vis);
                normalization_term += conv_weight;
                conv_u += params.conv_oversample;
            } //conv_u
            conv_v += params.conv_oversample;
            convolved_grid_v += params.nx;
        } //conv_v
    }
};
}