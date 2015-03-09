#pragma once

#include <stdexcept>
#include <complex>
#include <omp.h>
#include "polarization_gridding_traits.h"
#include "phase_transform_policies.h"
#include "jones_2x2.h"

namespace imaging {
  /**
   The polarization gridding policies define how the polarization terms are handled when being. It is for instance 
   possible to specify RR, LL + RR, XX + XY + YX + YY polarizations with Measurement Set v2.0. By swapping policies
   it should be easy to add / remove polarization terms (and / or corrective transformations) from the gridding step 
   of the pipeline and / or the distribution of computation.
   */
  template <typename visibility_base_type,typename uvw_base_type,
	    typename weights_base_type,typename convolution_base_type,typename grid_base_type,
	    typename phase_transformation_policy_type,typename gridding_mode>
  class polarization_gridding_policy {
  public:
    typedef polarization_gridding_trait<visibility_base_type,weights_base_type,float> trait_type;
    
    polarization_gridding_policy() { throw std::runtime_error("Undefined behaviour"); }
    /**
     Serves as proxy to future implementations. Subsequent headers should conform to the following:
     baseline_time_index: row index of the primary table of an MS v 2.0
     spw_index: current spectral window index
     channel_index: channel in current row
     uvw: coordinate of current row
     visibility: reference to visibility to read into and transform (if applicable)
     vis_weight_store: reference to a weight to store the flagged visibility weight to (this is used for normalization purposes)
     */
    inline void transform(std::size_t baseline_time_index, std::size_t spw_index, std::size_t channel_index, 
			  const uvw_coord<uvw_base_type> & __restrict__ uvw,
			  typename trait_type::pol_vis_type & __restrict__ visibility,
			  typename trait_type::pol_weight_type & __restrict__ vis_weight_store) __restrict__ {
      throw std::runtime_error("Undefined behaviour");
    }
    /**
     Serves as proxy to future implementations. Subsequent headers should conform to the following:
     term_flat_index: flat index of the individual polarization grids. Subsequent implementations should offset this index to the correct
		      grid automatically.
     visibility: reference to visibility being gridded
     convolution_weight: convolution weight to be applied to each polarization
    */
    inline void grid_polarization_terms(std::size_t term_flat_index, const typename trait_type::pol_vis_type & __restrict__ visibility,
					convolution_base_type convolution_weight) __restrict__ {
      throw std::runtime_error("Undefined behaviour");
    }
    /**
     Serves as proxy to future implementations. Subsequent headers should conform to the following:
     channel_grid_id: index of the channel grid to which is being accumulated
     facet_id: index of the facet being processed
     flagged_vis_weight: flagged vis weight as computed by transform
     sum_conv_weight: total convolution filter weight (accumulated accross (2N+1)**2 taps)
     */
    inline void update_sample_count_accumulator(std::size_t channel_grid_id, std::size_t facet_id, 
						typename trait_type::pol_vis_weight_type & __restrict__ flagged_vis_weight,
						convolution_base_type sum_conv_weight){
      throw std::runtime_error("Undefined behaviour");
    }
  };
  
  /**
   Policy to define gridding behaviour for a single polarization
   This policy makes provision for ignoring all but a single polarization.
   */
  template <typename visibility_base_type,typename uvw_base_type,
	    typename weights_base_type,typename convolution_base_type,typename grid_base_type,
	    typename phase_transformation_policy_type>
  class polarization_gridding_policy <visibility_base_type,uvw_base_type,weights_base_type,
				      convolution_base_type,grid_base_type,phase_transformation_policy_type,gridding_single_pol>{
  public:
      typedef polarization_gridding_trait<visibility_base_type,weights_base_type,gridding_single_pol> trait_type;
  protected:
      const phase_transformation_policy_type & __restrict__ _phase_transform_term;
      grid_base_type * __restrict__ _output_grids; //this is still complex but we want to reference the real and imag parts seperately in the atomics
      const typename trait_type::pol_vis_type * __restrict__ _visibilities;
      const typename trait_type::pol_vis_weight_type * __restrict__ _weights;
      const typename trait_type::pol_vis_flag_type * __restrict__ _flags;
      std::size_t _no_polarizations_in_data;
      std::size_t _polarization_index;
      std::size_t _channel_count;
      grid_base_type * __restrict__ _sample_counts;
      std::size_t _cube_channel_dim_size;
      std::size_t _num_facet_centres;
  public:
      /**
       Arguements:
       phase_transform_term: active phase transform policy to be applied to all polarizations
       output_grid: pointer to nx x ny pre-allocated buffer
       visibilities: set of complex visibility terms (flat-indexed: visibility[b x t][c][p] with the last index the fast-varying index)
       weights: set of weights to apply (per correlation) to each channel. This corresponds to the WEIGHT_SPECTRUM column. If this 
		column is not available the WEIGHT column gives the averages over all channels. In such an event each average should 
		be duplicated accross all the channels. It is however preferable to have a weight per channel. See for instance the 
		imaging chapter of Synthesis Imaging II.
       flags: flat-indexed boolean flagging array of dimensions timestamp_count x baseline_count x channel_count x polarization_term_count
       no_polarizations_in_data: this is the number of polarizations in the input data
       polarization_index: index of the polarization to grid
       channel_count: Number of channels per spectral window
      */
      polarization_gridding_policy(const phase_transformation_policy_type & phase_transform_term,
				   std::complex<grid_base_type> * output_grids,
				   const std::complex<visibility_base_type> * visibilities,
				   const weights_base_type * weights,
				   const bool* flags,
				   std::size_t no_polarizations_in_data,
				   std::size_t polarization_index,
				   std::size_t channel_count,
				   grid_base_type * sample_counts,
				   std::size_t cube_channel_dim_size,
				   std::size_t num_facet_centres
				  ):
				   _phase_transform_term(phase_transform_term), _output_grids((grid_base_type *)output_grids), 
				   _visibilities(visibilities),
				   _weights(weights), _flags(flags),
				   _no_polarizations_in_data(no_polarizations_in_data),
				   _polarization_index(polarization_index),
				   _channel_count(channel_count),
				   _sample_counts(sample_counts),
				   _cube_channel_dim_size(cube_channel_dim_size),
				   _num_facet_centres(num_facet_centres){}
      inline void transform(std::size_t baseline_time_index, std::size_t spw_index, std::size_t channel_index, 
			    const uvw_coord<uvw_base_type> & __restrict__ uvw,
			    typename trait_type::pol_vis_type & __restrict__ visibility,
			    typename trait_type::pol_vis_weight_type & __restrict__ vis_weight_store) __restrict__ {
	std::size_t visibility_flat_index = (baseline_time_index * _channel_count + channel_index) * _no_polarizations_in_data + _polarization_index;
	visibility = _visibilities[visibility_flat_index];
	/*
	 MS v2.0: weights are applied for each visibility term (baseline x channel x correlation)
	*/
	typename trait_type::pol_vis_weight_type weight = _weights[visibility_flat_index];
	typename trait_type::pol_vis_flag_type flag = _flags[visibility_flat_index];
	/*
	 do faceting phase shift (Cornwell & Perley, 1992) if enabled (through policy)
	 This faceting phase shift is a scalar matrix and commutes with everything (Smirnov, 2011), so 
	 we can apply it to the visibility immediately.
	*/
	_phase_transform_term.transform(visibility,uvw);
	vis_weight_store = weight * (int)(!flag);
	visibility *= vis_weight_store; //the integral promotion defines false == 0 and true == 1, this avoids unecessary branch divergence
      }
      inline void grid_polarization_terms(std::size_t term_flat_index, const typename trait_type::pol_vis_type & __restrict__ visibility,
					  convolution_base_type convolution_weight) __restrict__ {
	term_flat_index = term_flat_index << 1;
	#pragma omp atomic
	_output_grids[term_flat_index] += convolution_weight * visibility.real();
	#pragma omp atomic
	_output_grids[term_flat_index + 1] += convolution_weight * visibility.imag();
      }
      inline void update_sample_count_accumulator(std::size_t channel_grid_id, std::size_t facet_id, 
						typename trait_type::pol_vis_weight_type & __restrict__ flagged_vis_weight,
						convolution_base_type sum_conv_weight){
	_sample_counts[((omp_get_thread_num() * _num_facet_centres + facet_id) * 
			_cube_channel_dim_size + channel_grid_id) * 
		       1 + 0] += sum_conv_weight*flagged_vis_weight;
      }
  };
  
  /**
   Policy to define gridding behaviour for 2 user-specified polarizations
   This policy makes provision for computing I, Q, U, V stokes images from two polarizations
   */
  template <typename visibility_base_type,typename uvw_base_type,
	    typename weights_base_type,typename convolution_base_type,typename grid_base_type,
	    typename phase_transformation_policy_type>
  class polarization_gridding_policy <visibility_base_type,uvw_base_type,weights_base_type,
				      convolution_base_type,grid_base_type,phase_transformation_policy_type,gridding_double_pol>{
  public:
      typedef polarization_gridding_trait<visibility_base_type,weights_base_type,gridding_double_pol> trait_type;
  protected: 
      const phase_transformation_policy_type & __restrict__ _phase_transform_term;
      grid_base_type * __restrict__ _output_grids; //this is still complex but we want to reference the real and imag parts seperately in the atomics
      const std::complex<visibility_base_type> *  __restrict__ _visibilities;
      const weights_base_type * __restrict__ _weights;
      const bool * __restrict__ _flags;
      std::size_t _no_polarizations_in_data;
      std::size_t _first_polarization_index;
      std::size_t _second_polarization_index;
      std::size_t _grid_no_pixels;
      std::size_t _channel_count;
      grid_base_type * __restrict__ _sample_counts;
      std::size_t _cube_channel_dim_size;
      std::size_t _num_facet_centres;
  public:
      /**
       Arguements:
       phase_transform_term: active phase transform policy to be applied to all polarizations
       output_grid: pointer to nx x ny pre-allocated buffer
       visibilities: set of complex visibility terms (flat-indexed: visibility[b x t][c][p] with the last index the fast-varying index)
       weights: set of weights to apply (per correlation) to each channel. This corresponds to the WEIGHT_SPECTRUM column. If this 
		column is not available the WEIGHT column gives the averages over all channels. In such an event each average should 
		be duplicated accross all the channels. It is however preferable to have a weight per channel. See for instance the 
		imaging chapter of Synthesis Imaging II.
       flags: flat-indexed boolean flagging array of dimensions timestamp_count x baseline_count x channel_count x polarization_term_count
       no_polarizations_in_data: this is the number of polarizations in the input data
       first_polarization_index: index of the first polarization to grid
       second_polarization_index: index of the second polarization to grid
       channel_count: Number of channels per spectral window
      */
      polarization_gridding_policy(const phase_transformation_policy_type & phase_transform_term,
				   std::complex<grid_base_type> * output_grids,
				   const std::complex<visibility_base_type> * visibilities,
				   const weights_base_type * weights,
				   const bool * flags,
				   std::size_t no_polarizations_in_data,
				   std::size_t first_polarization_index,
				   std::size_t second_polarization_index,
				   std::size_t grid_no_pixels,
				   std::size_t channel_count,
				   grid_base_type * sample_counts,
				   std::size_t cube_channel_dim_size,
				   std::size_t num_facet_centres
				  ):
				   _phase_transform_term(phase_transform_term), _output_grids((grid_base_type *)output_grids), 
				   _visibilities(visibilities),
				   _weights(weights), 
				   _flags(flags),
				   _no_polarizations_in_data(no_polarizations_in_data),
				   _first_polarization_index(first_polarization_index),
				   _second_polarization_index(second_polarization_index),
				   _grid_no_pixels(grid_no_pixels),
				   _channel_count(channel_count),
				   _sample_counts(sample_counts),
				   _cube_channel_dim_size(cube_channel_dim_size),
				   _num_facet_centres(num_facet_centres){}
      inline void transform(std::size_t baseline_time_index, std::size_t spw_index, std::size_t channel_index, 
			    const uvw_coord<uvw_base_type> & __restrict__ uvw,
			    typename trait_type::pol_vis_type & __restrict__ visibility,
			    typename trait_type::pol_vis_weight_type & __restrict__ vis_weight_store) __restrict__ {
	//fetch four complex visibility terms from memory at a time:
	std::size_t visibility_flat_index = (baseline_time_index * _channel_count + channel_index) * _no_polarizations_in_data;
	visibility.v[0] = _visibilities[visibility_flat_index + _first_polarization_index]; 
	visibility.v[1] = _visibilities[visibility_flat_index + _second_polarization_index];
	/*
	 MS v2.0: weights are applied for each visibility term (baseline x channel x correlation)
	*/
	typename trait_type::pol_vis_weight_type weight = {_weights[visibility_flat_index + _first_polarization_index],
							   _weights[visibility_flat_index + _second_polarization_index]};
	typename trait_type::pol_vis_flag_type flag = {_flags[visibility_flat_index + _first_polarization_index],
						       _flags[visibility_flat_index + _second_polarization_index]};
	/*
	 do faceting phase shift (Cornwell & Perley, 1992) if enabled (through policy)
	 This faceting phase shift is a scalar matrix and commutes with everything (Smirnov, 2011), so 
	 we can apply it to the visibility immediately.
	*/
	_phase_transform_term.transform(visibility.v[0],uvw);
	_phase_transform_term.transform(visibility.v[1],uvw);
	vis_weight_store.w[0] = weight.w[0] * (int)(!flag.f[0]);
	vis_weight_store.w[1] = weight.w[1] * (int)(!flag.f[1]);
	visibility.v[0] *= vis_weight_store.w[0]; //the integral promotion defines false == 0 and true == 1, this avoids unecessary branch divergence
	visibility.v[1] *= vis_weight_store.w[1]; //the integral promotion defines false == 0 and true == 1, this avoids unecessary branch divergence
      }
      inline void grid_polarization_terms(std::size_t term_flat_index, const typename trait_type::pol_vis_type & __restrict__ visibility,
					  convolution_base_type convolution_weight) __restrict__ {
	std::size_t flat_indexed_reals_corr_1 = term_flat_index << 1;
	std::size_t flat_indexed_reals_corr_2 = (term_flat_index + _grid_no_pixels) << 1;
	#pragma omp atomic
	_output_grids[flat_indexed_reals_corr_1] += convolution_weight * visibility.v[0].real();
	#pragma omp atomic
	_output_grids[flat_indexed_reals_corr_1 + 1] += convolution_weight * visibility.v[0].imag();
	#pragma omp atomic
	_output_grids[flat_indexed_reals_corr_2] += convolution_weight * visibility.v[1].real();
	#pragma omp atomic
	_output_grids[flat_indexed_reals_corr_2 + 1] += convolution_weight * visibility.v[1].imag();
      }
      inline void update_sample_count_accumulator(std::size_t channel_grid_id, std::size_t facet_id, 
						typename trait_type::pol_vis_weight_type & __restrict__ flagged_vis_weight,
						convolution_base_type sum_conv_weight){
	std::size_t sample_count_flat_index = ((omp_get_thread_num() * _num_facet_centres + facet_id) * 
					      _cube_channel_dim_size + channel_grid_id) * 2;
	_sample_counts[sample_count_flat_index + 0] += sum_conv_weight*flagged_vis_weight.w[0];
	_sample_counts[sample_count_flat_index + 1] += sum_conv_weight*flagged_vis_weight.w[1];
      }
  };
  
  /**
   Policy to define gridding behaviour for 2x2 polarized visibilities 
   */
  template <typename visibility_base_type,typename uvw_base_type,
	    typename weights_base_type,typename convolution_base_type,typename grid_base_type,
	    typename phase_transformation_policy_type>
  class polarization_gridding_policy <visibility_base_type,uvw_base_type,weights_base_type,
				      convolution_base_type,grid_base_type,phase_transformation_policy_type,gridding_4_pol>{
  public:
      typedef polarization_gridding_trait<visibility_base_type,weights_base_type,gridding_4_pol> trait_type;
  protected:    
      const phase_transformation_policy_type & __restrict__ _phase_transform_term;
      grid_base_type * __restrict__ _output_grids; //this is still complex but we want to reference the real and imag parts seperately in the atomics
      const typename trait_type::pol_vis_type * __restrict__ _visibilities;
      const typename trait_type::pol_vis_weight_type * __restrict__ _weights;
      const typename trait_type::pol_vis_flag_type * __restrict__ _flags;
      std::size_t _grid_no_pixels;
      std::size_t _channel_count; 
      grid_base_type * __restrict__ _sample_counts;
      std::size_t _cube_channel_dim_size;
      std::size_t _num_facet_centres;
  public:
      /**
       Arguements:
       phase_transform_term: active phase transform policy to be applied to all polarizations
       output_grid: pointer to nx x ny pre-allocated buffer
       visibilities: set of complex visibility terms (flat-indexed: visibility[b x t][c][p] with the last index the fast-varying index)
       weights: set of weights to apply (per correlation) to each channel. This corresponds to the WEIGHT_SPECTRUM column. If this 
		column is not available the WEIGHT column gives the averages over all channels. In such an event each average should 
		be duplicated accross all the channels. It is however preferable to have a weight per channel. See for instance the 
		imaging chapter of Synthesis Imaging II.
       flags: flat-indexed boolean flagging array of dimensions timestamp_count x baseline_count x channel_count x polarization_term_count
       grid_no_pixels: total number of cells in the grid
       channel_count: Number of channels per spectral window
      */
      polarization_gridding_policy(const phase_transformation_policy_type & phase_transform_term,
				   std::complex<grid_base_type> * output_grids,
				   const std::complex<visibility_base_type> * visibilities,
				   const weights_base_type * weights,
				   const bool* flags,
				   std::size_t grid_no_pixels,
				   std::size_t channel_count,
				   grid_base_type * sample_counts,
				   std::size_t cube_channel_dim_size,
				   std::size_t num_facet_centres
				  ):
				   _phase_transform_term(phase_transform_term), _output_grids((grid_base_type *)output_grids), 
				   _visibilities((typename trait_type::pol_vis_type *)visibilities),
				   _weights((typename trait_type::pol_vis_weight_type *) weights), 
				   _flags((typename trait_type::pol_vis_flag_type *)flags),
				   _grid_no_pixels(grid_no_pixels),
				   _channel_count(channel_count),
				   _sample_counts(sample_counts),
				   _cube_channel_dim_size(cube_channel_dim_size),
				   _num_facet_centres(num_facet_centres){}
      __attribute__((optimize("unroll-loops")))
      inline void transform(std::size_t baseline_time_index, std::size_t spw_index, std::size_t channel_index, 
			    const uvw_coord<uvw_base_type> & __restrict__ uvw,
			    typename trait_type::pol_vis_type & __restrict__ visibility,
			    typename trait_type::pol_vis_weight_type & __restrict__ vis_weight_store) __restrict__ {
	//fetch four complex visibility terms from memory at a time:
	std::size_t visibility_jones_flat_index = baseline_time_index * _channel_count + channel_index;
	visibility = _visibilities[visibility_jones_flat_index];
	//MS v2.0: weights are applied per polarization, over all channels for each (baseline x time) step (row of the primary table):
	typename trait_type::pol_vis_weight_type weights = _weights[visibility_jones_flat_index];
	typename trait_type::pol_vis_flag_type flags = _flags[visibility_jones_flat_index];
	/*
	 do faceting phase shift (Cornwell & Perley, 1992) if enabled (through policy)
	 This faceting phase shift is a scalar matrix and commutes with everything (Smirnov, 2011), so 
	 we can apply it to the visibility immediately.
	*/
	for (std::size_t i = 0; i < 4; ++i){
	  _phase_transform_term.transform(visibility.correlations[i],uvw);
	  vis_weight_store.w[i] = weights.w[i] * (int)(!flags.f[i]);
	  visibility.correlations[i] *= vis_weight_store.w[i]; //the integral promotion defines false == 0 and true == 1, this avoids unecessary branch divergence
	}
      }
      __attribute__((optimize("unroll-loops")))
      inline void grid_polarization_terms(std::size_t term_flat_index, const typename trait_type::pol_vis_type & __restrict__ visibility,
					  convolution_base_type convolution_weight) __restrict__ {
	for (std::size_t i = 0; i < 4; ++i){
	  std::size_t grid_offset = i * _grid_no_pixels;
	  #pragma omp atomic
	  _output_grids[((grid_offset + term_flat_index) << 1)] += convolution_weight * visibility.correlations[i].real();
	  #pragma omp atomic
	  _output_grids[((grid_offset + term_flat_index) << 1) + 1] += convolution_weight * visibility.correlations[i].imag();
	}
      }
      __attribute__((optimize("unroll-loops")))
      inline void update_sample_count_accumulator(std::size_t channel_grid_id, std::size_t facet_id, 
						typename trait_type::pol_vis_weight_type & __restrict__ flagged_vis_weight,
						convolution_base_type sum_conv_weight){
	std::size_t sample_count_flat_index = ((omp_get_thread_num() * _num_facet_centres + facet_id) * 
					      _cube_channel_dim_size + channel_grid_id) * 4;
	for (std::size_t i = 0; i < 4; ++i){				    
	  _sample_counts[sample_count_flat_index + i] += sum_conv_weight*flagged_vis_weight.w[i];
	}
      }
  };
  
  /**
   * Policy to enable applying corrective Jones terms (DDEs and Gain terms) for 4 correlations
   * 
   * This policy is simply an extension to the basic 4 correlation policy and adds the 
   * Jones "union model" transformations to the 2x2 correlated visibility terms
   * See Smirnov I (2011) for details, note that for facet-based gridding we can assume the 
   * slow-varying DDE terms can be corrected for outside the integral - Smirnov II (2011)
   */
  template <typename visibility_base_type,typename uvw_base_type,
	    typename weights_base_type,typename convolution_base_type,typename grid_base_type,
	    typename phase_transformation_policy_type>
  class polarization_gridding_policy <visibility_base_type,uvw_base_type,weights_base_type,
				      convolution_base_type,grid_base_type,phase_transformation_policy_type,
				      gridding_4_pol_enable_facet_based_jones_corrections> : 
				      public polarization_gridding_policy <visibility_base_type,uvw_base_type,weights_base_type,
				      convolution_base_type,grid_base_type,phase_transformation_policy_type,gridding_4_pol>{
  public:
      typedef polarization_gridding_trait<visibility_base_type,weights_base_type,gridding_4_pol_enable_facet_based_jones_corrections> trait_type;
  protected:  
      const jones_2x2<visibility_base_type> * __restrict__ _inverted_jones_terms;
      std::size_t _direction_index;
      std::size_t _direction_count;
      std::size_t _antenna_count;
      std::size_t _calibration_timestamp_count;
      std::size_t _spw_count;
      const unsigned int *  __restrict__  _antenna_1_ids;
      const unsigned int *  __restrict__  _antenna_2_ids;
      const std::size_t *  __restrict__  _timestamp_ids;
      grid_base_type * __restrict__ _sample_counts;
  public:
      /**
       Arguments:
       phase_transform_term: active phase transform policy to be applied to all polarizations
       output_grid: pointer to nx x ny pre-allocated buffer
       visibilities: set of complex visibility terms (flat-indexed: visibility[b x t][c][p] with the last index the fast-varying index)
       weights: set of weights to apply (per correlation) to each channel. This corresponds to the WEIGHT_SPECTRUM column. If this 
		column is not available the WEIGHT column gives the averages over all channels. In such an event each average should 
		be duplicated accross all the channels. It is however preferable to have a weight per channel. See for instance the 
		imaging chapter of Synthesis Imaging II.
       flags: flat-indexed boolean flagging array of dimensions timestamp_count x baseline_count x channel_count x polarization_term_count
       grid_no_pixels: total number of cells in the grid
       channel_count: Number of channels per spectral window
       inverted_jones_matricies: Set of jones matricies of dimensions [time x antenna count x direction count x spectral window count x channel count x 4 correlations].
				 Must be in indexed in this order. Uses the property (J^-1)^H == (J^H)^-1 to significantly decrease number of computations
				 while gridding. 
       antenna_1_ids: First antennae ids (per measurement set row)
       antenna_2_ids: Second antennae ids (per measurement set row)
       timestamp_ids: Row timestamp id, measurement set time stamp id. This id corresponds to the time dimension of the inverted jones matricies.
       antenna_count: Number of antennae in measurement set
       direction_index: Facet index (it is assumed there is a computed DDE jones term per facet)
       direction_count: Number of directions being faceted
       calibration_timestamp_count: Number of timestamps being imaged. The time dimension of the set of inverted jones matricies is assumed equal to this
				    number.
       spw_count: number of spectral windows being imaged. It is assumed the inverted jones matricies will be computed per channel in each spectral window,
		  since the DDEs vary per frequency.
      */
      polarization_gridding_policy(const phase_transformation_policy_type & phase_transform_term,
				   std::complex<grid_base_type> * __restrict__ output_grids,
				   const std::complex<visibility_base_type> * __restrict__ visibilities,
				   const weights_base_type * __restrict__ weights,
				   const bool* __restrict__ flags,
				   std::size_t grid_no_pixels,
				   std::size_t channel_count,
				   const jones_2x2<visibility_base_type> * __restrict__ inverted_jones_terms,
				   const unsigned int *  __restrict__  antenna_1_ids,
				   const unsigned int *  __restrict__  antenna_2_ids,
				   const std::size_t *  __restrict__  timestamp_ids,
				   std::size_t antenna_count, std::size_t direction_index, 
				   std::size_t direction_count, std::size_t calibration_timestamp_count, 
				   std::size_t spw_count,
				   grid_base_type * sample_counts,
				   std::size_t cube_channel_dim_size,
				   std::size_t num_facet_centres
				  ):
				   _inverted_jones_terms(inverted_jones_terms), _direction_index(direction_index), 
				   _direction_count(direction_count), _antenna_count(antenna_count),
				   _antenna_1_ids(antenna_1_ids), _antenna_2_ids(antenna_2_ids),
				   _timestamp_ids(timestamp_ids), _calibration_timestamp_count(calibration_timestamp_count), 
				   _spw_count(spw_count),
				   polarization_gridding_policy<visibility_base_type,uvw_base_type,weights_base_type,
								convolution_base_type,grid_base_type,
								phase_transformation_policy_type,gridding_4_pol>(
								  phase_transform_term,output_grids,
								  visibilities,weights,flags,grid_no_pixels,
								  channel_count,sample_counts,
								  cube_channel_dim_size,num_facet_centres){}
      __attribute__((optimize("unroll-loops")))
      inline void transform(std::size_t baseline_time_index, std::size_t spw_index, std::size_t channel_index, 
			    const uvw_coord<uvw_base_type> & __restrict__ uvw,
			    typename trait_type::pol_vis_type & __restrict__ visibility,
			    typename trait_type::pol_vis_weight_type & __restrict__ vis_weight_store) __restrict__ {
	//fetch four complex visibility terms from memory at a time:
	std::size_t visibility_jones_flat_index = baseline_time_index * this->_channel_count + channel_index;
	visibility = this->_visibilities[visibility_jones_flat_index];
	//MS v2.0: weights are applied per polarization, over all channels for each (baseline x time) step (row of the primary table):
	typename trait_type::pol_vis_weight_type weights = this->_weights[visibility_jones_flat_index];
	typename trait_type::pol_vis_flag_type flags = this->_flags[visibility_jones_flat_index];
	/*
	 do faceting phase shift (Cornwell & Perley, 1992) if enabled (through policy)
	 This faceting phase shift is a scalar matrix and commutes with everything (Smirnov, 2011), so 
	 we can apply it to the visibility immediately.
	*/
	for (std::size_t i = 0; i < 4; ++i){
	  this->_phase_transform_term.transform(visibility.correlations[i],uvw);
	  vis_weight_store.w[i] = weights.w[i] * (int)(!flags.f[i]);
	  visibility.correlations[i] *= vis_weight_store.w[i]; //the integral promotion defines false == 0 and true == 1, this avoids unecessary branch divergence
	}
	/*
	 fetch antenna 1 and antenna 2 jones matricies (flat-indexed). Assume there is a Jones matrix per 
	 antenna per baseline, time, spw and channel as described in Smirnov I (2011). We may assume that the slow-varying
	 directional dependent effects are corrected for in this step as described in Smirnov II (2011).
	*/
	std::size_t antenna_1_id = _antenna_1_ids[baseline_time_index];
	std::size_t antenna_2_id = _antenna_2_ids[baseline_time_index];
	std::size_t time_index = _timestamp_ids[baseline_time_index];
	std::size_t antenna_1_jones_terms_flat_index = (((time_index*_antenna_count + antenna_1_id)*_direction_count + _direction_index)*_spw_count + spw_index)*this->_channel_count + channel_index;
	std::size_t antenna_2_jones_terms_flat_index = (((time_index*_antenna_count + antenna_2_id)*_direction_count + _direction_index)*_spw_count + spw_index)*this->_channel_count + channel_index;
	
	jones_2x2<visibility_base_type> antenna_1_jones = _inverted_jones_terms[antenna_1_jones_terms_flat_index];
	jones_2x2<visibility_base_type> antenna_2_jones = _inverted_jones_terms[antenna_2_jones_terms_flat_index];
	/*
	 gridding will invert the computed / observed jones terms on both sides: Jp^-1 x (V := Jp x B x Jq^H) x (Jq^-1)^H
	 by taking a normal inner product with complex terms
	 
	 Note that: (A^-1)^H == (A^H)^-1. We use this property to invert the matricies (per antenna) before even thinking of gridding. 
	 This will save us a lot of computation later on. We will only need to take the Hermitian transpose to avoid essentially, doubling 
	 the storage requirements to keep these matricies in memory
	*/
	
	do_hermitian_transpose<visibility_base_type>(antenna_2_jones);
	jones_2x2<visibility_base_type> tmp;
	inner_product<visibility_base_type>(visibility,antenna_2_jones,tmp);
	inner_product<visibility_base_type>(antenna_1_jones,tmp,visibility);
      }
  };
  
  /**
   Policy to define gridding behaviour for gridding the sampling function
   The sampling function is the same accross all correlation terms, but
   varies with channel (since u,v,w depends on wavelength)
   The sampling function is always normalized so that the centre pixel
   is 1 (Bill Cotton)
   */
  template <typename visibility_base_type,typename uvw_base_type,
	    typename weights_base_type,typename convolution_base_type,typename grid_base_type,
	    typename phase_transformation_policy_type>
  class polarization_gridding_policy <visibility_base_type,uvw_base_type,weights_base_type,
				      convolution_base_type,grid_base_type,phase_transformation_policy_type,gridding_sampling_function>{
  public:
      typedef polarization_gridding_trait<visibility_base_type,weights_base_type,gridding_sampling_function> trait_type;
  protected:
      const phase_transformation_policy_type & __restrict__ _phase_transform_term;
      grid_base_type * __restrict__ _output_grids; //this is still complex but we want to reference the real and imag parts seperately in the atomics
      const typename trait_type::pol_vis_flag_type * __restrict__ _flags;
      const typename trait_type::pol_vis_weight_type * __restrict__ _weights;
      std::size_t _no_polarizations_in_data;
      std::size_t _polarization_index;
      std::size_t _channel_count;
      std::size_t _cube_channel_dim_size;
      std::size_t _num_facet_centres;
  public:
      /**
       Arguements:
       phase_transform_term: active phase transform policy to be applied to all polarizations
       output_grid: pointer to nx x ny pre-allocated buffer
       flags: flat-indexed boolean flagging array of dimensions timestamp_count x baseline_count x channel_count x polarization_term_count
       weights: flat-indexed weights array of dimensions timestamp_count x baseline_count x channel_count x polarization_term_count
       no_polarizations_in_data: this is the number of polarizations in the input data
       polarization_index: index of the polarization to grid
       channel_count: Number of channels per spectral window
      */
      polarization_gridding_policy(const phase_transformation_policy_type & phase_transform_term,
				   std::complex<grid_base_type> * output_grids,
				   const bool* flags,
				   const weights_base_type * weights,
				   std::size_t no_polarizations_in_data,
				   std::size_t polarization_index,
				   std::size_t channel_count
				  ):
				   _phase_transform_term(phase_transform_term), 
				   _output_grids((grid_base_type *)output_grids), 
				   _flags(flags),
				   _weights(weights),
				   _no_polarizations_in_data(no_polarizations_in_data),
				   _polarization_index(polarization_index),
				   _channel_count(channel_count){}
      inline void transform(std::size_t baseline_time_index, std::size_t spw_index, std::size_t channel_index, 
			    const uvw_coord<uvw_base_type> & __restrict__ uvw,
			    typename trait_type::pol_vis_type & __restrict__ visibility,
			    typename trait_type::pol_vis_weight_type & __restrict__ vis_weight_store) __restrict__ {
	//fetch four complex visibility terms from memory at a time:
	std::size_t visibility_flat_index = (baseline_time_index * _channel_count + channel_index) * _no_polarizations_in_data + _polarization_index;
	visibility = 1;
	/*
	 MS v2.0: weights are applied for each visibility term (baseline x channel x correlation)
	*/
	typename trait_type::pol_vis_flag_type flag = _flags[visibility_flat_index];
	typename trait_type::pol_vis_weight_type weight = _weights[visibility_flat_index];
	/*
	 do faceting phase shift (Cornwell & Perley, 1992) if enabled (through policy)
	 This faceting phase shift is a scalar matrix and commutes with everything (Smirnov, 2011), so 
	 we can apply it to the visibility immediately.
	*/
	vis_weight_store = weight * (int)(!flag);
	visibility *= vis_weight_store; //the integral promotion defines false == 0 and true == 1, this avoids unecessary branch divergence
      }
      inline void grid_polarization_terms(std::size_t term_flat_index, const typename trait_type::pol_vis_type & __restrict__ visibility,
					  convolution_base_type convolution_weight) __restrict__ {
	term_flat_index = term_flat_index << 1;
	#pragma omp atomic
	_output_grids[term_flat_index] += convolution_weight * visibility;
      }
      inline void update_sample_count_accumulator(std::size_t channel_grid_id, std::size_t facet_id, 
						typename trait_type::pol_vis_weight_type & __restrict__ flagged_vis_weight,
						convolution_base_type sum_conv_weight){
	//The PSF is normalized according to the centre pixel (should be unity according to Bill Cotton) and 
	//therefore this method is left blank
      }
  };
}