#pragma once

#include <stdexcept>
#include <complex>
#include "phase_transform_policies.h"
#include "jones_2x2.h"

namespace imaging {
  class gridding_single_pol {};
  class gridding_4_pol {};
  class gridding_4_pol_enable_facet_based_jones_corrections {};
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
    polarization_gridding_policy() { throw std::exception("Undefined behaviour"); }
    /**
     Serves as proxy to future implementations. Subsequent headers should conform to the following:
     baseline_time_index: row index of the primary table of an MS v 2.0
     channel_index
     baseline_count
     timestamp_count
     channel_count
     */
    inline void transform(std::size_t baseline_time_index, std::size_t channel_index, std::size_t baseline_count, 
			    std::size_t timestamp_count, std::size_t channel_count, const uvw_coord<uvw_base_type> & uvw) __restrict__ {
      throw std::exception("Undefined behaviour");
    }
    /**
     Serves as proxy to future implementations. Subsequent headers should conform to the following:
     term_flat_index: flat index of the individual polarization grids. Subsequent implementations should offset this index to the correct
		      grid automatically.
     grid_no_pixels: number of pixels per polarization grid (nx * ny)
     convolution_weight: convolution weight to be applied to each polarization
    */
    inline void grid_polarization_terms(std::size_t term_flat_index, std::size_t grid_no_pixels, convolution_base_type convolution_weight) __restrict__ {
      throw std::exception("Undefined behaviour");
    }
  };
  
  template <typename weights_base_type>
  struct visibility_weights_2x2 {
    weights_base_type _weights[4];
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
  protected:
      const phase_transformation_policy_type & __restrict__ _phase_transform_term;
      std::complex<grid_base_type> * __restrict__ _output_grids;
      const std::complex<visibility_base_type> * __restrict__ _visibilities;
      const weights_base_type * __restrict__ _weights;
      const bool * __restrict__ _flags;
      std::complex<visibility_base_type> _visibility;
      std::size_t _no_polarizations_in_data;
      std::size_t _polarization_index;
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
      */
      polarization_gridding_policy(const phase_transformation_policy_type & phase_transform_term,
				   std::complex<grid_base_type> * output_grids,
				   const std::complex<visibility_base_type> * visibilities,
				   const weights_base_type * weights,
				   const bool* flags,
				   std::size_t no_polarizations_in_data,
				   std::size_t polarization_index):
				   _phase_transform_term(phase_transform_term), _output_grids(output_grids), 
				   _visibilities(visibilities),
				   _weights(weights), _flags(flags),
				   _no_polarizations_in_data(no_polarizations_in_data),_polarization_index(polarization_index){}
      inline void transform(std::size_t baseline_time_index, std::size_t channel_index, std::size_t baseline_count, 
			    std::size_t timestamp_count, std::size_t channel_count, const uvw_coord<uvw_base_type> & uvw) __restrict__ {
	//fetch four complex visibility terms from memory at a time:
	std::size_t visibility_flat_index = (baseline_time_index * channel_count + channel_index) * _no_polarizations_in_data + _polarization_index;
	_visibility = _visibilities[visibility_flat_index];
	/*
	 MS v2.0: weights are applied for each visibility term (baseline x channel x correlation)
	*/
	weights_base_type weight = _weights[visibility_flat_index];
	bool flag = _flags[visibility_flat_index];
	/*
	 do faceting phase shift (Cornwell & Perley, 1992) if enabled (through policy)
	 This faceting phase shift is a scalar matrix and commutes with everything (Smirnov, 2011), so 
	 we can apply it to the visibility immediately.
	*/
	_phase_transform_term.transform(_visibility,uvw);
	_visibility *= weight * (int)(!flag); //the integral promotion defines false == 0 and true == 1, this avoids unecessary branch divergence
      }
      inline void grid_polarization_terms(std::size_t term_flat_index, std::size_t grid_no_pixels, convolution_base_type convolution_weight) __restrict__ {
	_output_grids[term_flat_index] += convolution_weight * _visibility;
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
  protected:
      const phase_transformation_policy_type & __restrict__ _phase_transform_term;
      std::complex<grid_base_type> * __restrict__ _output_grids;
      const jones_2x2<visibility_base_type> * __restrict__ _visibilities;
      const visibility_weights_2x2<weights_base_type> * __restrict__ _weights;
      const visibility_weights_2x2<bool>* __restrict__ _flags;
      jones_2x2<visibility_base_type> _visibility_polarizations;
      jones_2x2<visibility_base_type> _conj_visibility_polarizations;
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
      */
      polarization_gridding_policy(const phase_transformation_policy_type & phase_transform_term,
				   std::complex<grid_base_type> * output_grids,
				   const std::complex<visibility_base_type> * visibilities,
				   const weights_base_type * weights,
				   const bool* flags):
				   _phase_transform_term(phase_transform_term), _output_grids(output_grids), 
				   _visibilities((jones_2x2<visibility_base_type> *)visibilities),
				   _weights((visibility_weights_2x2<weights_base_type> *) weights), _flags((visibility_weights_2x2<bool>*)flags){}
      __attribute__((optimize("unroll-loops")))
      inline void transform(std::size_t baseline_time_index, std::size_t channel_index, std::size_t baseline_count, 
			    std::size_t timestamp_count, std::size_t channel_count, const uvw_coord<uvw_base_type> & uvw) __restrict__ {
	//fetch four complex visibility terms from memory at a time:
	std::size_t visibility_jones_flat_index = baseline_time_index * channel_count + channel_index;
	_visibility_polarizations = _visibilities[visibility_jones_flat_index];
	//MS v2.0: weights are applied per polarization, over all channels for each (baseline x time) step (row of the primary table):
	visibility_weights_2x2<weights_base_type> weights = _weights[visibility_jones_flat_index];
	visibility_weights_2x2<bool> flags = _flags[visibility_jones_flat_index];
	/*
	 do faceting phase shift (Cornwell & Perley, 1992) if enabled (through policy)
	 This faceting phase shift is a scalar matrix and commutes with everything (Smirnov, 2011), so 
	 we can apply it to the visibility immediately.
	*/
	for (std::size_t i = 0; i < 4; ++i){
	  _phase_transform_term.transform(_visibility_polarizations._polarizations[i],uvw);
	  _visibility_polarizations._polarizations[i] *= weights._weights[i] * (int)(!flags._weights[i]); //the integral promotion defines false == 0 and true == 1, this avoids unecessary branch divergence
	  _conj_visibility_polarizations._polarizations[i] = conj(_visibility_polarizations._polarizations[i]);
	}
      }
      __attribute__((optimize("unroll-loops")))
      inline void grid_polarization_terms(std::size_t term_flat_index, std::size_t grid_no_pixels, convolution_base_type convolution_weight) __restrict__ {
	for (std::size_t i = 0; i < 4; ++i){
	  std::size_t grid_offset = i * grid_no_pixels;
	  _output_grids[grid_offset + term_flat_index] += convolution_weight * _visibility_polarizations._polarizations[i];
	}
      }
  };
  
  /**
   * Policy to enable applying corrective Jones terms (DDEs and Gain terms) for 4 polarizations
   * 
   * This policy is simply an extension to the basic 4 polarization policy and adds the 
   * Jones "union model" transformations to the 2x2 polarized visibility terms
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
  protected:
      const jones_2x2<visibility_base_type> * __restrict__ _jones_terms;
      std::size_t _direction_index;
      std::size_t _direction_count;
      std::size_t _antenna_count;
      const std::size_t *  __restrict__  _antenna_1_ids;
      const std::size_t *  __restrict__  _antenna_2_ids;
      jones_2x2<visibility_base_type> _true_visibility_polarizations;
  public:
      polarization_gridding_policy(const phase_transformation_policy_type & phase_transform_term,
				   std::complex<grid_base_type> * __restrict__ output_grids,
				   const std::complex<visibility_base_type> * __restrict__ visibilities,
				   const weights_base_type * __restrict__ weights,
				   const bool* __restrict__ flags,
				   const jones_2x2<visibility_base_type> * __restrict__ jones_terms,
				   const std::size_t *  __restrict__  antenna_1_ids,
				   const std::size_t *  __restrict__  antenna_2_ids,
				   std::size_t antenna_count, std::size_t direction_index, 
				   std::size_t direction_count):
				   _jones_terms(jones_terms), _direction_index(direction_index), 
				   _direction_count(direction_count), _antenna_count(antenna_count),
				   _antenna_1_ids(antenna_1_ids), _antenna_2_ids(antenna_2_ids),
				   polarization_gridding_policy<visibility_base_type,uvw_base_type,weights_base_type,
								convolution_base_type,grid_base_type,
								phase_transformation_policy_type,gridding_4_pol>
							(phase_transform_term,output_grids,
							 visibilities,weights,flags){}
				      
      inline void transform(std::size_t baseline_time_index, std::size_t channel_index, std::size_t baseline_count, 
			    std::size_t timestamp_count, std::size_t channel_count, const uvw_coord<uvw_base_type> & uvw) __restrict__ {
	/*
	 Set up everything just as in a 2x2 polarized gridding (inherited behaviour), but then add on the Jones correction terms
	 */
	polarization_gridding_policy<visibility_base_type,uvw_base_type,weights_base_type,convolution_base_type,grid_base_type,
				     phase_transformation_policy_type,gridding_4_pol>::transform(baseline_time_index, channel_index, 
												 baseline_count, timestamp_count, 
												 channel_count,uvw);
	/*
	 fetch antenna 1 and antenna 2 jones matricies (flat-indexed). Assume there is a Jones matrix per 
	 antenna per baseline, time and channel as described in Smirnov I (2011). We may assume that the slow-varying
	 directional dependent effects are corrected for in this step as described in Smirnov II (2011).
	*/
	std::size_t antenna_1_id = _antenna_1_ids[baseline_time_index];
	std::size_t antenna_2_id = _antenna_2_ids[baseline_time_index];
	std::size_t time_index = baseline_time_index / baseline_count;
	std::size_t antenna_1_jones_terms_flat_index = ((antenna_1_id*timestamp_count + time_index)*channel_count + 
						       channel_index)*_direction_count + _direction_index;
	std::size_t antenna_2_jones_terms_flat_index = ((antenna_2_id*timestamp_count + time_index)*channel_count + 
						       channel_index)*_direction_count + _direction_index;
	jones_2x2<visibility_base_type> antenna_1_jones = _jones_terms[antenna_1_jones_terms_flat_index];
	jones_2x2<visibility_base_type> antenna_2_jones = _jones_terms[antenna_2_jones_terms_flat_index];
	
	/*
	 gridding inverts the computed / observed jones terms on both sides: Jp^-1 x (V := Jp x B x Jq^H) x (Jq^H)^-1
	 by taking a normal inner product with complex terms
	*/
	jones_2x2<visibility_base_type>::do_inverse(antenna_1_jones);
	jones_2x2<visibility_base_type>::do_hermitian_transpose(antenna_2_jones);
	jones_2x2<visibility_base_type>::do_inverse(antenna_2_jones);
	jones_2x2<visibility_base_type> tmp;
	jones_2x2<visibility_base_type>::inner_product(this->_visibility_polarizations,antenna_2_jones,tmp);
	jones_2x2<visibility_base_type>::inner_product(tmp,antenna_2_jones,_true_visibility_polarizations);
      }
  };
}