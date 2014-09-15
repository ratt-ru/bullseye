#include <cassert>
#include <cstdlib>
#include <cstdio>
#include <complex>
#include <memory>
#include <algorithm>
#include <random>
#include <chrono>

#include "uvw_coord.h"
#include "baseline_transform_policies.h"
#include "phase_transform_policies.h"
#include "polarization_gridding_policies.h"
#include "convolution_policies.h"
#include "gridding.h"

int main (int argc, char ** argv) {
    using namespace std;
    using namespace imaging;
    typedef float visibility_base_type;
    typedef float uvw_base_type;
    typedef float reference_wavelengths_base_type;
    typedef float convolution_base_type;
    typedef float visibility_weights_base_type;
    typedef float grid_base_type;

    assert(argc == 8);
    size_t row_count = atol(argv[1]);
    size_t nx = atol(argv[2]);
    size_t ny = atol(argv[3]);
    size_t chan_no = atol(argv[4]);
    size_t pol_count = atol(argv[5]);
    size_t conv_support = atol(argv[6]);
    size_t conv_oversample = atol(argv[7]);
    reference_wavelengths_base_type ref_wavelength = 0.245;
    float cell_size_l = 2;
    float cell_size_m = 2;
    size_t no_timestamps = row_count;
    size_t no_baselines = 1;

    {
        printf("ALLOCATING MEMORY FOR %ld x %ld COMPLEX GRID\n",nx,ny);
        std::unique_ptr<complex<grid_base_type> > output_buffer(new complex<grid_base_type>[nx*ny]);
        printf("ALLOCATING MEMORY FOR %ld ROWS (%ld chan, %ld pol EACH)\n",row_count,chan_no,pol_count);
        std::unique_ptr<complex<visibility_base_type> > visibilities(new complex<visibility_base_type>[row_count*chan_no*pol_count]);
        std::unique_ptr<visibility_weights_base_type> visibility_weights(new visibility_weights_base_type[row_count*chan_no*pol_count]);
        std::unique_ptr<bool> flags(new bool[row_count*chan_no*pol_count]()); //init all to false
        std::unique_ptr<uvw_coord<uvw_base_type> > uvw_coords(new uvw_coord<uvw_base_type>[row_count]);
        
	/*
	 * randomly initialize the uvw coordinates uniformly:
	 */
	std::mt19937 r_eng;
	std::chrono::high_resolution_clock myclock;
	r_eng.seed(myclock.now().time_since_epoch().count());
	std::uniform_real_distribution<uvw_base_type> rand_dist((uvw_base_type)(-min(nx,ny)/2*ref_wavelength),
									(uvw_base_type)(min(nx,ny)/2*ref_wavelength));
	auto rand = [&](){ return rand_dist(r_eng); };
	std::generate(uvw_coords.get(),uvw_coords.get() + chan_no, 
		      [rand](){return rand();});
	
        std::unique_ptr<bool> flagged_rows(new bool[row_count]()); //init all to false
        std::unique_ptr<reference_wavelengths_base_type> reference_wavelengths(new reference_wavelengths_base_type[chan_no]);
        std::generate(reference_wavelengths.get(),reference_wavelengths.get() + chan_no, [ref_wavelength]() {
            return ref_wavelength;
        });
        std::unique_ptr<unsigned int > field_array(new unsigned int[row_count]()); //init all to 0
        std::unique_ptr<unsigned int > spw_array(new unsigned int[row_count]()); //init all to 0
        printf("ALLOCATING MEMORY FOR CONVOLUTION KERNEL WITH %ld CELL SUPPORT, OVERSAMPLED BY FACTOR OF %ld\n",conv_support,conv_oversample);
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
        printf("BENCHMARK TERMINATED SUCCESSFULLY\n");
    }
    return 0;
}