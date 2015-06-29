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
#include <cassert>
#include <cstdlib>
#include <cstdio>
#include <complex>
#include <memory>
#include <algorithm>
#include <random>
#include <chrono>
#include <cmath>
#include <iostream>
#include <omp.h>
#include <stdexcept>
#include "uvw_coord.h"
#include "wrapper.h"
#include "gridding_parameters.h"
#include "base_types.h"


const double TO_GIB = 1.0/(1024.0*1024.0*1024.0);

using namespace std;
using namespace imaging;

/**
 * We'll simulate a EVLA 27 antenna setup here. Coordinates taken from an actual measurement set of an
 * observation of G55.7+3.4_10s.
 */
const size_t NO_ANTENNAE = 27;
const size_t NO_BASELINES = (NO_ANTENNAE * (NO_ANTENNAE - 1))/2 + NO_ANTENNAE;
const uvw_coord<uvw_base_type> antenna_coords[] = {{-1601710.017000f , -5042006.925200f , 3554602.355600f},
    {-1601150.060300f , -5042000.619800f , 3554860.729400f},
    {-1600715.950800f , -5042273.187000f , 3554668.184500f},
    {-1601189.030140f , -5042000.493300f , 3554843.425700f},
    {-1601614.091000f , -5042001.652900f , 3554652.509300f},
    {-1601162.591000f , -5041828.999000f , 3555095.896400f},
    {-1601014.462000f , -5042086.252000f , 3554800.799800f},
    {-1601185.634945f , -5041978.156586f , 3554876.424700f},
    {-1600951.588000f , -5042125.911000f , 3554773.012300f},
    {-1601177.376760f , -5041925.073200f , 3554954.584100f},
    {-1601068.790300f , -5042051.910200f , 3554824.835300f},
    {-1600801.926000f , -5042219.366500f , 3554706.448200f},
    {-1601155.635800f , -5041783.843800f , 3555162.374100f},
    {-1601447.198000f , -5041992.502500f , 3554739.687600f},
    {-1601225.255200f , -5041980.383590f , 3554855.675000f},
    {-1601526.387300f , -5041996.840100f , 3554698.327400f},
    {-1601139.485100f , -5041679.036800f , 3555316.533200f},
    {-1601315.893000f , -5041985.320170f , 3554808.304600f},
    {-1601168.786100f , -5041869.054000f , 3555036.936000f},
    {-1601192.467800f , -5042022.856800f , 3554810.438800f},
    {-1601173.979400f , -5041902.657700f , 3554987.517500f},
    {-1600880.571400f , -5042170.388000f , 3554741.457400f},
    {-1601377.009500f , -5041988.665500f , 3554776.393400f},
    {-1601180.861480f , -5041947.453400f , 3554921.628700f},
    {-1601265.153600f , -5041982.533050f , 3554834.858400f},
    {-1601114.365500f , -5042023.151800f , 3554844.944000f},
    {-1601147.940400f , -5041733.837000f , 3555235.956000f}
};

int main (int argc, char ** argv) {
    if (argc != 14)
        throw runtime_error("Expected args num_threads,dataset_(int)_size_in_MiB,nx,ny,num_chans,num_corr,conv_half_support_size,conv_times_oversample,num_wplanes,observation_length_in_hours,ra_0,dec_0,num_facets");
    size_t no_threads = atol(argv[1]);
    size_t dataset_size = atol(argv[2]);
    size_t nx = atol(argv[3]);
    size_t ny = atol(argv[4]);
    size_t chan_no = atol(argv[5]);
    size_t pol_count = atol(argv[6]);
    if (pol_count != 1 && pol_count != 2 && pol_count != 4)
      throw std::invalid_argument("Expected 1,2 or 4 correlations in argument");
    size_t conv_support = atol(argv[7]);
    size_t conv_oversample = atol(argv[8]);
    size_t num_wplanes = atol(argv[9]);
    if (num_wplanes <= 1) printf("WARNING: DISABLING WPROJECTION CONVOLUTIONS\n");
    size_t conv_full_support = conv_support * 2 + 1;
    size_t conv_padded_full_support = conv_full_support + 2;
    size_t convolution_size_dim = conv_padded_full_support + (conv_padded_full_support - 1) * (conv_oversample - 1);
    size_t convolution_slice_size = convolution_size_dim * convolution_size_dim;
    size_t convolution_cube_size = convolution_slice_size * num_wplanes;
    printf("------------------------------------\nGRIDDER BENCHMARK\n(USING %ld THREADS)\n------------------------------------\n",no_threads);
    omp_set_num_threads((int)no_threads);
    gridding_parameters params;
    size_t row_size = sizeof(bool)*chan_no*4+
		      sizeof(visibility_weights_base_type)*chan_no*4+
		      sizeof(complex<visibility_base_type>)*chan_no*4+
		      sizeof(uvw_coord<uvw_base_type>)+
		      sizeof(bool)+
		      sizeof(unsigned int)+
		      sizeof(unsigned int);
    size_t no_timestamps = (dataset_size * 1024 * 1024) / (row_size * NO_BASELINES);
    size_t row_count = no_timestamps * NO_BASELINES;
    float hrs = atof(argv[10]);
    float time_step = (hrs / 24.0 * 2 * M_PI) / no_timestamps; //time step in radians
    float ra = atof(argv[11]) * M_PI / 180;
    float declination = atof(argv[12]) * M_PI / 180;
    size_t num_facets = atol(argv[13]);
    if (num_facets == 0) printf("WARNING: DISABLING FACETING\n");
    void (*gridding_function)(gridding_parameters &) = num_facets == 0 ? ((pol_count == 1) ? grid_single_pol : (pol_count == 2) ? grid_duel_pol : grid_4_cor) :
									 ((pol_count == 1) ? facet_single_pol : (pol_count == 2) ? facet_duel_pol : facet_4_cor);
    std::unique_ptr<uvw_base_type[]> facet_centre_list(new uvw_base_type[num_facets*2]);
    for (size_t f = 0; f < num_facets*2; f += 2){
      facet_centre_list.get()[f] = ra;
      facet_centre_list.get()[f+1] = declination;
    }
    reference_wavelengths_base_type ref_wavelength = 0.245;
    float cell_size_l = 2;
    float cell_size_m = 2;
    printf("ALLOCATING MEMORY FOR %ld ROWS (%ld chan, %d pol EACH) (%f GiB)\n",row_count,chan_no,4,
           row_size*row_count*TO_GIB);
    std::unique_ptr<complex<visibility_base_type>[] > visibilities(new complex<visibility_base_type>[row_count*chan_no*4]);
    std::unique_ptr<visibility_weights_base_type[]> visibility_weights(new visibility_weights_base_type[row_count*chan_no*4]);
    std::unique_ptr<bool[]> flags(new bool[row_count*chan_no*4]()); //init all to false
    std::unique_ptr<bool[]> flagged_rows(new bool[row_count]()); //init all to false
    std::unique_ptr<reference_wavelengths_base_type[]> reference_wavelengths(new reference_wavelengths_base_type[chan_no]);
    std::generate(reference_wavelengths.get(),reference_wavelengths.get() + chan_no, [ref_wavelength]() {
        return ref_wavelength;
    });
    std::unique_ptr<unsigned int[] > field_array(new unsigned int[row_count]()); //init all to 0
    std::unique_ptr<unsigned int[] > spw_array(new unsigned int[row_count]()); //init all to 0
    std::unique_ptr<std::size_t[] > chan_grid_indicies(new std::size_t[chan_no*1]()); //init all to 0
    std::unique_ptr<bool[] > enabled_chans(new bool[chan_no*1]);
    std::generate(enabled_chans.get(),enabled_chans.get() + chan_no*1, []() {
        return true;
    });
    /**
     * This generates uvw coordinates based on antenna positions, number of hours observed and declination of the simulated telescope
     * Synthesis Imaging II, pp 25-26, except that we're rotating in the opposite direction here Hour angle vs right ascension
     *
     * This may not conform perfectly to the measurement set uvw coordinates, but they act simularly to those found in a ms in the
     * sence that the rotation of the earth sweep out eliptical uv paths through a plane parellel to the rotation of the earth (ie. the
     * coordinates paths are circular when viewed at NCP
     */
    std::unique_ptr<uvw_coord<uvw_base_type>[] > uvw_coords(new uvw_coord<uvw_base_type>[row_count]);
    printf("COMPUTING UVW COORDINATES\n");
    std::generate(uvw_coords.get(),uvw_coords.get() + row_count,
    [ra,declination,time_step]() {
        static size_t row;
        static size_t l = NO_ANTENNAE;
        static size_t k = NO_ANTENNAE;
        size_t timestamp = row / (NO_BASELINES);
        size_t baseline_index = row % (NO_BASELINES);
        size_t increment_antenna_1_coord = (baseline_index / k);
        /*
         * calculate antenna 1 and antenna 2 ids based on baseline index using some fancy
         * footwork ;). This indexing scheme will enumerate all unique baselines per
         * timestamp.
         */
        l -= (1) * increment_antenna_1_coord;
        k += (l) * increment_antenna_1_coord;
        size_t antenna_1 = NO_ANTENNAE-l;
        size_t antenna_2 = NO_ANTENNAE + (baseline_index-k);
        size_t new_timestamp = ((baseline_index+1) / NO_BASELINES);
        k -= (NO_BASELINES-NO_ANTENNAE) * new_timestamp;
        l += (NO_ANTENNAE-1) * new_timestamp;
        uvw_base_type Lx = antenna_coords[antenna_1]._u - antenna_coords[antenna_2]._u;
        uvw_base_type Ly = antenna_coords[antenna_1]._v - antenna_coords[antenna_2]._v;
        uvw_base_type Lz = antenna_coords[antenna_1]._w - antenna_coords[antenna_2]._w;
        uvw_base_type rotation_in_radians = timestamp*time_step + ra;
        uvw_base_type sin_ra = sin(rotation_in_radians);
        uvw_base_type cos_ra = cos(rotation_in_radians);
        uvw_base_type sin_dec = sin(declination);
        uvw_base_type cos_dec = cos(declination);
        uvw_base_type u = -sin_ra*Lx + cos_ra*Ly;
        uvw_base_type v = -sin_dec*cos_ra*Lx - sin_dec*sin_ra*Ly + cos_dec*Lz;
        uvw_base_type w = cos_dec*cos_ra*Lx + cos_dec*sin_ra*Ly + sin_dec*Lz;
        ++row;
        return uvw_coord<uvw_base_type>(u,v,w);
    });
    printf("ALLOCATING MEMORY FOR %ld x %ld COMPLEX GRID FOR %ld FACETS (%f GiB)\n",nx,ny,num_facets,std::max<size_t>(1,num_facets)*pol_count*nx*ny*sizeof(complex<grid_base_type>)*TO_GIB);
    std::unique_ptr<complex<grid_base_type>[] > output_buffer(new complex<grid_base_type>[std::max<size_t>(1,num_facets)*pol_count*nx*ny]());
    params.antenna_count = NO_ANTENNAE;
    params.baseline_count = NO_BASELINES;
    params.cell_size_x = cell_size_l;
    params.cell_size_y = cell_size_m;
    params.num_facet_centres = num_facets;
    params.facet_centres = facet_centre_list.get();
    params.channel_count = chan_no;
    params.channel_grid_indicies = chan_grid_indicies.get();
    params.cube_channel_dim_size = 1;
    params.conv_oversample = conv_oversample;
    params.conv_support = conv_support;
    params.cube_channel_dim_size = 1;
    params.enabled_channels = enabled_chans.get();
    params.field_array = field_array.get();
    params.flagged_rows = flagged_rows.get();
    params.flags = flags.get();
    params.imaging_field = 0;
    params.no_timestamps_read = no_timestamps;
    params.num_facet_centres = std::max<size_t>(1,num_facets);
    params.number_of_polarization_terms = pol_count;
    params.nx = nx;
    params.ny = ny;
    params.output_buffer = output_buffer.get();
    params.phase_centre_dec = declination;
    params.phase_centre_ra = ra;
    params.polarization_index = 0;
    params.second_polarization_index = 2;
    params.number_of_polarization_terms = 4;
    params.number_of_polarization_terms_being_gridded = pol_count;
    params.reference_wavelengths = reference_wavelengths.get();
    params.row_count = row_count;
    params.spw_count = 1;
    params.spw_index_array = spw_array.get();
    params.uvw_coords = uvw_coords.get();
    params.visibilities = visibilities.get();
    params.visibility_weights = visibility_weights.get();
    params.wplanes = num_wplanes;
    params.wmax_est = 6500;
    if (num_wplanes > 1){
      printf("ALLOCATING MEMORY FOR %ld CONVOLUTION KERNELS WITH %ld CELL SUPPORT, OVERSAMPLED BY FACTOR OF %ld (%f GiB)\n",
	    num_wplanes,conv_support,conv_oversample,convolution_cube_size*sizeof(std::complex<convolution_base_type>)*TO_GIB);
      std::unique_ptr<std::complex<convolution_base_type>[] > conv(new std::complex<convolution_base_type>[convolution_cube_size]);
      #pragma omp parallel for
      for (size_t w = 0; w < num_wplanes; ++w){
	size_t w_offset = w * convolution_slice_size;
	for (size_t cv = 0; cv < conv_full_support; ++cv){
	  double PI_X = M_PI * (cv / (convolution_base_type)(conv_oversample)) - ((convolution_base_type) conv_support);
	  convolution_base_type sinc_v = (PI_X != 0) ? sin(PI_X) / PI_X : 1.0;
	  for (size_t cu = 0; cu < conv_full_support; ++cu){
	    double PI_X = M_PI * (cu / (convolution_base_type)(conv_oversample)) - ((convolution_base_type) conv_support);
	    convolution_base_type sinc_u = (PI_X != 0) ? sin(PI_X) / PI_X : 1.0;
	    conv.get()[w_offset + cv * convolution_size_dim + cu] = std::complex<convolution_base_type>(sinc_u * sinc_v,0);
	  }
	}
      }
      params.conv = (convolution_base_type *)(conv.get());
      initLibrary(params);
      gridding_function(params);  
      releaseLibrary();
    } else {
      printf("ALLOCATING MEMORY FOR %ld CONVOLUTION KERNELS WITH %ld CELL SUPPORT, OVERSAMPLED BY FACTOR OF %ld (%f GiB)\n",
	    num_wplanes,conv_support,conv_oversample,convolution_cube_size/convolution_size_dim*sizeof(convolution_base_type)*TO_GIB);
      std::unique_ptr<convolution_base_type[]> conv(new convolution_base_type[convolution_cube_size/convolution_size_dim]);
      #pragma omp parallel for
      for (size_t w = 0; w < num_wplanes; ++w){
	size_t w_offset = w * convolution_slice_size;
	for (size_t x = 0; x < conv_full_support; ++x){
	  double PI_X = M_PI * (x / (convolution_base_type)(conv_oversample)) - ((convolution_base_type) conv_support);
	  convolution_base_type sinc = (PI_X != 0) ? sin(PI_X) / PI_X : 1.0;
	  conv.get()[w_offset + x] = sinc;
	}
      }
      params.conv = (convolution_base_type *)(conv.get());
      initLibrary(params);
      gridding_function(params);  
      releaseLibrary();
    }
        
    printf("COMPUTE COMPLETED IN %f SECONDS\n",get_gridding_walltime());
    printf("BENCHMARK TERMINATED SUCCESSFULLY\n");
    return 0;
}