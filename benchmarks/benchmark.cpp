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
#include "timer.h"
#include "uvw_coord.h"
#include "baseline_transform_policies.h"
#include "phase_transform_policies.h"
#include "polarization_gridding_policies.h"
#include "convolution_policies.h"
#include "gridding.h"
const double TO_GIB = 1.0/(1024.0*1024.0*1024.0);

using namespace std;
using namespace imaging;
typedef float visibility_base_type;
typedef float uvw_base_type;
typedef float reference_wavelengths_base_type;
typedef float convolution_base_type;
typedef float visibility_weights_base_type;
typedef float grid_base_type;    

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
						   {-1601147.940400f , -5041733.837000f , 3555235.956000f}};

int main (int argc, char ** argv) {
    if (argc != 12)
      throw runtime_error("Expected args num_threads,num_timestamps,nx,ny,num_chans,num_pols,conv_support_size,conv_times_oversample,observation_length_in_hours,ra_0,dec_0");
    size_t no_threads = atol(argv[1]);
    size_t no_timestamps = atol(argv[2]);
    size_t row_count = no_timestamps * NO_BASELINES;
    size_t nx = atol(argv[3]);
    size_t ny = atol(argv[4]);
    size_t chan_no = atol(argv[5]);
    size_t pol_count = atol(argv[6]);
    size_t conv_support = atol(argv[7]);
    size_t conv_oversample = atol(argv[8]);
    float hrs = atof(argv[9]);
    float time_step = (hrs / 24.0 * 2 * M_PI) / no_timestamps; //time step in radians
    float ra = atof(argv[10]) * M_PI / 180;
    float declination = atof(argv[11]) * M_PI / 180;
    
    reference_wavelengths_base_type ref_wavelength = 0.245;
    float cell_size_l = 2;
    float cell_size_m = 2;

    {
	printf("------------------------------------\nGRIDDER BENCHMARK\n(USING %ld THREADS)\n------------------------------------\n",no_threads);
	omp_set_num_threads((int)no_threads);
        printf("ALLOCATING MEMORY FOR %ld x %ld COMPLEX GRID (%f GiB)\n",nx,ny,nx*ny*sizeof(grid_base_type)*TO_GIB);
        std::unique_ptr<complex<grid_base_type> > output_buffer(new complex<grid_base_type>[nx*ny]);
        printf("ALLOCATING MEMORY FOR %ld ROWS (%ld chan, %ld pol EACH) (%f GiB)\n",row_count,chan_no,pol_count,
	       (sizeof(bool)+
		sizeof(visibility_weights_base_type)*chan_no*pol_count+
		sizeof(complex<visibility_base_type>)*chan_no*pol_count+
		sizeof(uvw_coord<uvw_base_type>)+
		sizeof(bool)+
		sizeof(unsigned int)+
		sizeof(unsigned int))*row_count*TO_GIB);
        std::unique_ptr<complex<visibility_base_type> > visibilities(new complex<visibility_base_type>[row_count*chan_no*pol_count]);
        std::unique_ptr<visibility_weights_base_type> visibility_weights(new visibility_weights_base_type[row_count*chan_no*pol_count]);
        std::unique_ptr<bool> flags(new bool[row_count*chan_no*pol_count]()); //init all to false
        std::unique_ptr<uvw_coord<uvw_base_type> > uvw_coords(new uvw_coord<uvw_base_type>[row_count]);
	/**
	 * This generates uvw coordinates based on antenna positions, number of hours observed and declination of the simulated telescope
	 * Synthesis Imaging II, pp 25-26, except that we're rotating in the opposite direction here Hour angle vs right ascension
	 * 
	 * This may not conform perfectly to the measurement set uvw coordinates, but they act simularly to those found in a ms in the
	 * sence that the rotation of the earth sweep out eliptical uv paths through a plane parellel to the rotation of the earth (ie. the
	 * coordinates paths are circular when viewed at NCP
	 */
	printf("COMPUTING UVW COORDINATES\n");
	std::generate(uvw_coords.get(),uvw_coords.get() + row_count, 
		      [ra,declination,time_step](){
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
			float Lx = antenna_coords[antenna_1]._u - antenna_coords[antenna_2]._u;
			float Ly = antenna_coords[antenna_1]._v - antenna_coords[antenna_2]._v;
			float Lz = antenna_coords[antenna_1]._w - antenna_coords[antenna_2]._w;
			float rotation_in_radians = timestamp*time_step + ra;
			float sin_ra = sin(rotation_in_radians);
			float cos_ra = cos(rotation_in_radians);
			float sin_dec = sin(declination);
			float cos_dec = cos(declination);
			float u = -sin_ra*Lx + cos_ra*Ly;
			float v = -sin_dec*cos_ra*Lx - sin_dec*sin_ra*Ly + cos_dec*Lz;
			float w = cos_dec*cos_ra*Lx + cos_dec*sin_ra*Ly + sin_dec*Lz;
			++row;
			return uvw_coord<uvw_base_type>(u,v,w);
		      });
	
        std::unique_ptr<bool> flagged_rows(new bool[row_count]()); //init all to false
        std::unique_ptr<reference_wavelengths_base_type> reference_wavelengths(new reference_wavelengths_base_type[chan_no]);
        std::generate(reference_wavelengths.get(),reference_wavelengths.get() + chan_no, [ref_wavelength]() {
            return ref_wavelength;
        });
        std::unique_ptr<unsigned int > field_array(new unsigned int[row_count]()); //init all to 0
        std::unique_ptr<unsigned int > spw_array(new unsigned int[row_count]()); //init all to 0
        printf("ALLOCATING MEMORY FOR CONVOLUTION KERNEL WITH %ld CELL SUPPORT, OVERSAMPLED BY FACTOR OF %ld (%f GiB)\n",
	       conv_support,conv_oversample,conv_support*conv_oversample*sizeof(convolution_base_type)*TO_GIB);
        std::unique_ptr<convolution_base_type> conv(new convolution_base_type[conv_oversample*conv_support*conv_oversample*conv_support]);
        std::generate(conv.get(),conv.get() + conv_oversample*conv_support*conv_oversample*conv_support, [ref_wavelength]() {
            return 1;
        });
	std::unique_ptr<std::size_t> chan_grid_indicies(new std::size_t[chan_no*1]()); //init all to 0
	std::unique_ptr<bool> enabled_chans(new bool[chan_no*1]);
	std::generate(enabled_chans.get(),enabled_chans.get() + chan_no*1, []() {
            return true;
        });
        typedef baseline_transform_policy<uvw_base_type, transform_disable_facet_rotation> baseline_transform_policy_type;
        typedef phase_transform_policy<visibility_base_type,
                uvw_base_type,
                transform_disable_phase_rotation> phase_transform_policy_type;
        typedef polarization_gridding_policy<visibility_base_type, uvw_base_type,
                visibility_weights_base_type, convolution_base_type, grid_base_type,
                phase_transform_policy_type, gridding_single_pol> polarization_gridding_policy_type;
        typedef convolution_policy<convolution_base_type,uvw_base_type,grid_base_type,
                polarization_gridding_policy_type, convolution_precomputed_fir> convolution_policy_type;

        baseline_transform_policy_type uvw_transform; 	 //standard: no uvw rotation
        phase_transform_policy_type phase_transform; //standard: no phase rotation

        polarization_gridding_policy_type polarization_policy(phase_transform,
                output_buffer.get(),
                visibilities.get(),
                visibility_weights.get(),
                flags.get(),
                pol_count,
                0,
                chan_no);
        convolution_policy_type convolution_policy(nx,ny,pol_count,conv_support,conv_oversample,
                conv.get(), polarization_policy);
	utils::timer walltime;
	walltime.start();
        imaging::grid<visibility_base_type,uvw_base_type,
                reference_wavelengths_base_type,convolution_base_type,
                visibility_weights_base_type,grid_base_type,
                baseline_transform_policy_type,
                polarization_gridding_policy_type,
                convolution_policy_type>
                (polarization_policy,uvw_transform,convolution_policy,
                 uvw_coords.get(),
                 flagged_rows.get(),
                 nx,ny,
                 casa::Quantity(cell_size_l,"arcsec"),
                 casa::Quantity(cell_size_m,"arcsec"),
                 chan_no,row_count,
                 reference_wavelengths.get(),
                 field_array.get(),
                 0,
                 spw_array.get(),
		 chan_grid_indicies.get(),
		 enabled_chans.get());
	walltime.stop();
        printf("COMPUTE COMPLETED IN %f SECONDS\n",walltime.duration());
    }
    printf("BENCHMARK TERMINATED SUCCESSFULLY\n");
    return 0;
}