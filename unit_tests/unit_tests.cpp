#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
#include "catch.hpp"

#include <complex>
#include <cmath>
#include <casa/Quanta/Quantum.h>

#include "phase_transform_policies.h"
#include "baseline_transform_policies.h"
#include "polarization_gridding_policies.h"
#include "jones_2x2.h"
#include "convolution_policies.h"

using namespace imaging;
using namespace std;

bool is_close(double val, double expect){
  #define EPSILON 0.0000001
  return abs(expect - val) <= EPSILON;
}
TEST_CASE( "Testing uvw coordinate operators" ) {
  SECTION( "Testing scalar operator*=" ){
    uvw_coord<double> uvw(2,5,6);
    uvw *= 0.5;
    REQUIRE(is_close(uvw._u,1.0));
    REQUIRE(is_close(uvw._v,2.5));
    REQUIRE(is_close(uvw._w,3.0));
  }
  SECTION( "Testing vector operator*=" ){
    uvw_coord<double> uvw(2,5,6);
    uvw *= uvw_coord<double>(0.5,2,3);
    REQUIRE(is_close(uvw._u,1.0));
    REQUIRE(is_close(uvw._v,10));
    REQUIRE(is_close(uvw._w,18));
  }
  SECTION( "Testing operator-" ){
    uvw_coord<double> uvw(2,5,6);
    uvw_coord<double> uvw_neg = -uvw;
    REQUIRE(is_close(uvw._u,2.0));
    REQUIRE(is_close(uvw._v,5.0));
    REQUIRE(is_close(uvw._w,6.0));
    REQUIRE(is_close(uvw_neg._u,-2.0));
    REQUIRE(is_close(uvw_neg._v,-5.0));
    REQUIRE(is_close(uvw_neg._w,-6.0));
  }
}
TEST_CASE( "Testing phase transform policies" ) {
  SECTION( "Testing the base policy" ) {
    typedef phase_transform_policy<float,float,float> policy_type;
    REQUIRE_THROWS(policy_type a_policy);
  }
  SECTION( "Testing the disabled phase transform" ){
    phase_transform_policy<float,float,transform_disable_phase_rotation> a_policy;
    complex<float> visibility(3.0f,5.0f);
    uvw_coord<float> uvw(1.5f,6.0f,12.0f);
    a_policy.transform(visibility,uvw);
    REQUIRE(visibility.real() == 3.0f);
    REQUIRE(visibility.imag() == 5.0f);
    REQUIRE(uvw._u == 1.5f);
    REQUIRE(uvw._v == 6.0f);
    REQUIRE(uvw._w == 12.0f);
  }
  SECTION( "Testing the enabled left-handed ra dec phase transform" ){
    phase_transform_policy<double,double,transform_enable_phase_rotation_lefthanded_ra_dec> a_policy(casa::Quantity(126000,"arcsec"),casa::Quantity(144000,"arcsec"),
												      casa::Quantity(100000,"arcsec"),casa::Quantity(135000,"arcsec"));
    complex<double> visibility(3.0f,5.0f);
    uvw_coord<double> uvw(1.5f,6.0f,12.0f);
    a_policy.transform(visibility,uvw);
    REQUIRE(is_close(visibility.real(),-1.069499681));
    REQUIRE(is_close(visibility.imag(),-5.732030219));
    //no modification
    REQUIRE(uvw._u == 1.5f);
    REQUIRE(uvw._v == 6.0f);
    REQUIRE(uvw._w == 12.0f);
  }
}
TEST_CASE( "Testing baseline transform policies" ) {
  SECTION( "Testing the base policy" ) {
    typedef baseline_transform_policy<float,float> policy_type;
    REQUIRE_THROWS(policy_type a_policy);
  }
  SECTION( "Testing the disabled uvw transform" ){
    baseline_transform_policy <double, transform_disable_facet_rotation> a_policy;
    uvw_coord<double> uvw(1.5f,6.0f,12.0f);
    a_policy.transform(uvw);
    REQUIRE(is_close(uvw._u,1.5f));
    REQUIRE(is_close(uvw._v,6.0f));
    REQUIRE(is_close(uvw._w,12.0f));
  }
  SECTION( "Testing the left-handed uvw transform" ){
    baseline_transform_policy <double, transform_facet_lefthanded_ra_dec> a_policy(0,0,casa::Quantity(126000,"arcsec"),casa::Quantity(144000,"arcsec"),
										    casa::Quantity(100000,"arcsec"),casa::Quantity(135000,"arcsec"));
    uvw_coord<double> uvw(1.5f,6.0f,12.0f);
    a_policy.transform(uvw);
    REQUIRE(is_close(uvw._u,2.158906181));
    REQUIRE(is_close(uvw._v,6.658291832));
    REQUIRE(is_close(uvw._w,11.54366813));
  }
}

TEST_CASE( "Testing Jones Matrix operations" ) {  
  SECTION( "Testing the Hermitian transpose" ) {
    jones_2x2<float> j = {complex<float>(2,3),complex<float>(4,5),complex<float>(1,6),complex<float>(7,8)};
    do_hermitian_transpose<float>(j);
    REQUIRE(is_close(j.correlations[0].real(),2));
    REQUIRE(is_close(j.correlations[0].imag(),-3));
    REQUIRE(is_close(j.correlations[1].real(),1));
    REQUIRE(is_close(j.correlations[1].imag(),-6));
    REQUIRE(is_close(j.correlations[2].real(),4));
    REQUIRE(is_close(j.correlations[2].imag(),-5));
    REQUIRE(is_close(j.correlations[3].real(),7));
    REQUIRE(is_close(j.correlations[3].imag(),-8));
  }
  SECTION( "Testing the determinant" ) {
    jones_2x2<float> j = {complex<float>(2,3),complex<float>(4,5),complex<float>(1,6),complex<float>(7,8)};
    complex<float> detj = det<float>(j);
    REQUIRE(is_close(detj.real(),16));
    REQUIRE(is_close(detj.imag(),8));
    REQUIRE(is_close(j.correlations[0].real(),2));
    REQUIRE(is_close(j.correlations[0].imag(),3));
    REQUIRE(is_close(j.correlations[1].real(),4));
    REQUIRE(is_close(j.correlations[1].imag(),5));
    REQUIRE(is_close(j.correlations[2].real(),1));
    REQUIRE(is_close(j.correlations[2].imag(),6));
    REQUIRE(is_close(j.correlations[3].real(),7));
    REQUIRE(is_close(j.correlations[3].imag(),8));
  }
  SECTION( "Testing the inverse" ) {
    jones_2x2<float> j = {complex<float>(2,3),complex<float>(4,5),complex<float>(1,6),complex<float>(7,8)};
    invert(j);
    REQUIRE(is_close(j.correlations[0].real(),0.55));
    REQUIRE(is_close(j.correlations[0].imag(),0.225));
    REQUIRE(is_close(j.correlations[1].real(),-0.325));
    REQUIRE(is_close(j.correlations[1].imag(),-0.15));
    REQUIRE(is_close(j.correlations[2].real(),-0.2));
    REQUIRE(is_close(j.correlations[2].imag(),-0.275));
    REQUIRE(is_close(j.correlations[3].real(),0.175));
    REQUIRE(is_close(j.correlations[3].imag(),0.1));
  }
  SECTION( "Testing the all inverse" ) {
    jones_2x2<float> jones_set[] = {{complex<float>(2,3),complex<float>(4,5),complex<float>(1,6),complex<float>(7,8)},
				    {complex<float>(4,6),complex<float>(8,10),complex<float>(2,12),complex<float>(14,16)}};
    invert_all((jones_2x2<float> *)jones_set,2);
    REQUIRE(is_close(jones_set[0].correlations[0].real(),0.55));
    REQUIRE(is_close(jones_set[0].correlations[0].imag(),0.225));
    REQUIRE(is_close(jones_set[0].correlations[1].real(),-0.325));
    REQUIRE(is_close(jones_set[0].correlations[1].imag(),-0.15));
    REQUIRE(is_close(jones_set[0].correlations[2].real(),-0.2));
    REQUIRE(is_close(jones_set[0].correlations[2].imag(),-0.275));
    REQUIRE(is_close(jones_set[0].correlations[3].real(),0.175));
    REQUIRE(is_close(jones_set[0].correlations[3].imag(),0.1));
    
    REQUIRE(is_close(jones_set[1].correlations[0].real(),0.275));
    REQUIRE(is_close(jones_set[1].correlations[0].imag(),0.1125));
    REQUIRE(is_close(jones_set[1].correlations[1].real(),-0.1625));
    REQUIRE(is_close(jones_set[1].correlations[1].imag(),-0.075));
    REQUIRE(is_close(jones_set[1].correlations[2].real(),-0.1));
    REQUIRE(is_close(jones_set[1].correlations[2].imag(),-0.1375));
    REQUIRE(is_close(jones_set[1].correlations[3].real(),0.0875));
    REQUIRE(is_close(jones_set[1].correlations[3].imag(),0.05));
  }
  SECTION( "Testing the all inverse throws exception on inverting singular matrix" ) {
    jones_2x2<float> jones_set[] = {{complex<float>(2,3),complex<float>(4,5),complex<float>(1,6),complex<float>(7,8)},
				    {complex<float>(0,0),complex<float>(0,0),complex<float>(0,0),complex<float>(0,0)}};
    REQUIRE_THROWS(invert_all((jones_2x2<float> *)jones_set,2));
  }
  SECTION( "Testing the unrolled multiplication" ) {
    jones_2x2<float> a = {complex<float>(1,2),complex<float>(3,4),complex<float>(5,6),complex<float>(7,8)};
    jones_2x2<float> b = {complex<float>(9,10),complex<float>(11,12),complex<float>(13,14),complex<float>(15,16)};
    jones_2x2<float> c;
    inner_product<float>(a,b,c);
    REQUIRE(is_close(c.correlations[0].real(),-28));
    REQUIRE(is_close(c.correlations[0].imag(),122));
    REQUIRE(is_close(c.correlations[1].real(),-32));
    REQUIRE(is_close(c.correlations[1].imag(),142));
    REQUIRE(is_close(c.correlations[2].real(),-36));
    REQUIRE(is_close(c.correlations[2].imag(),306));
    REQUIRE(is_close(c.correlations[3].real(),-40));
    REQUIRE(is_close(c.correlations[3].imag(),358));
    
    REQUIRE(is_close(a.correlations[0].real(),1));
    REQUIRE(is_close(a.correlations[0].imag(),2));
    REQUIRE(is_close(a.correlations[1].real(),3));
    REQUIRE(is_close(a.correlations[1].imag(),4));
    REQUIRE(is_close(a.correlations[2].real(),5));
    REQUIRE(is_close(a.correlations[2].imag(),6));
    REQUIRE(is_close(a.correlations[3].real(),7));
    REQUIRE(is_close(a.correlations[3].imag(),8));
    
    REQUIRE(is_close(b.correlations[0].real(),9));
    REQUIRE(is_close(b.correlations[0].imag(),10));
    REQUIRE(is_close(b.correlations[1].real(),11));
    REQUIRE(is_close(b.correlations[1].imag(),12));
    REQUIRE(is_close(b.correlations[2].real(),13));
    REQUIRE(is_close(b.correlations[2].imag(),14));
    REQUIRE(is_close(b.correlations[3].real(),15));
    REQUIRE(is_close(b.correlations[3].imag(),16));
  }
  SECTION( "Testing the unrolled inplace-safe multiplication" ) {
    jones_2x2<float> a = {complex<float>(1,2),complex<float>(3,4),complex<float>(5,6),complex<float>(7,8)};
    jones_2x2<float> b = {complex<float>(9,10),complex<float>(11,12),complex<float>(13,14),complex<float>(15,16)};
    inner_product<float>(a,b,a);
    REQUIRE(is_close(a.correlations[0].real(),-28));
    REQUIRE(is_close(a.correlations[0].imag(),122));
    REQUIRE(is_close(a.correlations[1].real(),-32));
    REQUIRE(is_close(a.correlations[1].imag(),142));
    REQUIRE(is_close(a.correlations[2].real(),-36));
    REQUIRE(is_close(a.correlations[2].imag(),306));
    REQUIRE(is_close(a.correlations[3].real(),-40));
    REQUIRE(is_close(a.correlations[3].imag(),358));
    
    REQUIRE(is_close(b.correlations[0].real(),9));
    REQUIRE(is_close(b.correlations[0].imag(),10));
    REQUIRE(is_close(b.correlations[1].real(),11));
    REQUIRE(is_close(b.correlations[1].imag(),12));
    REQUIRE(is_close(b.correlations[2].real(),13));
    REQUIRE(is_close(b.correlations[2].imag(),14));
    REQUIRE(is_close(b.correlations[3].real(),15));
    REQUIRE(is_close(b.correlations[3].imag(),16));
  }
}
TEST_CASE( "Testing polarization handling policies" ){
  SECTION( "Testing the base policy" ){
    typedef polarization_gridding_policy<float,float,float,float,float,float,double> policy_type;
    REQUIRE_THROWS(policy_type a_policy);
  }
  SECTION( "Testing the single polarization policy" ){
    typedef float visibility_base_type;
    typedef float uvw_base_type;
    typedef float reference_wavelengths_base_type;
    typedef float convolution_base_type;
    typedef float visibility_weights_base_type;
    typedef float grid_base_type;
    
    typedef baseline_transform_policy<uvw_base_type, transform_disable_facet_rotation> baseline_transform_policy_type;
    typedef phase_transform_policy<visibility_base_type, uvw_base_type, transform_disable_phase_rotation> phase_transform_policy_type;
    typedef polarization_gridding_policy<visibility_base_type, uvw_base_type, visibility_weights_base_type, convolution_base_type, grid_base_type,
					 phase_transform_policy_type, gridding_single_pol> polarization_gridding_policy_type;
    baseline_transform_policy_type uvw_transform; //standard: no uvw rotation
    phase_transform_policy_type phase_transform; //standard: no phase rotation
    //emulate two rows in the measurement set
    size_t rows = 2;
    size_t ch = 2;
    size_t pols = 2;
    size_t nx = 50;
    size_t ny = 50;
    complex<grid_base_type> uv_grid[nx*ny];
    //<first_row>ch1:pol1,ch1:pol2,ch2:pol1,ch2:pol2,<second_row>ch1:pol1,ch1:pol2,ch2:pol1,ch2:pol2
    complex<visibility_base_type> vis[] = {complex<visibility_base_type>(0,1),complex<visibility_base_type>(1,0),
					   complex<visibility_base_type>(0,1),complex<visibility_base_type>(1,0),
					   complex<visibility_base_type>(0,1),complex<visibility_base_type>(1,0),
					   complex<visibility_base_type>(0,1),complex<visibility_base_type>(1,0)};
    visibility_weights_base_type weights[] = {1,2,3,4,5,6,7,8};
    bool flags[] = {false,false,false,true,false,false,true,false};
    uvw_coord<uvw_base_type> uvw(1,2,3);
    unsigned int spw [] = {0,0};
    polarization_gridding_policy_type polarization_policy(phase_transform,
							  (complex<grid_base_type>*)(uv_grid),
							  (complex<visibility_base_type>*)(vis),
							  (visibility_weights_base_type*)weights,
							  (bool*)flags,
							  pols,0,ch);
    for (size_t r = 0; r < rows; ++r){
	for (size_t c = 0; c < ch; ++c){
	  typename polarization_gridding_policy_type::trait_type::pol_vis_type vis;
	  typename polarization_gridding_policy_type::trait_type::pol_vis_type conj;
	  polarization_policy.transform(r,spw[r],c,uvw,vis);
	  polarization_policy.grid_polarization_terms(25*nx+25,vis,1);
	}
    }
    REQUIRE(uvw._u == 1);
    REQUIRE(uvw._v == 2);
    REQUIRE(uvw._w == 3);
    //grid all the even numbers 0,2,4,6 (pol 0), weighted 1,3,5,7, but the last one is flagged
    REQUIRE(is_close(uv_grid[25*nx+25].imag(),9));
  }
  SECTION( "Testing the 4 correlation term polarization policy" ){
    typedef float visibility_base_type;
    typedef float uvw_base_type;
    typedef float reference_wavelengths_base_type;
    typedef float convolution_base_type;
    typedef float visibility_weights_base_type;
    typedef float grid_base_type;
    
    typedef baseline_transform_policy<uvw_base_type, transform_disable_facet_rotation> baseline_transform_policy_type;
    typedef phase_transform_policy<visibility_base_type, uvw_base_type, transform_disable_phase_rotation> phase_transform_policy_type;
    typedef polarization_gridding_policy<visibility_base_type, uvw_base_type, visibility_weights_base_type, convolution_base_type, grid_base_type,
					 phase_transform_policy_type, gridding_4_pol> polarization_gridding_policy_type;
    baseline_transform_policy_type uvw_transform; //standard: no uvw rotation
    phase_transform_policy_type phase_transform; //standard: no phase rotation
    //emulate two rows in the measurement set
    size_t rows = 2;
    size_t ch = 2;
    size_t pols = 4;
    size_t nx = 50;
    size_t ny = 50;
    complex<grid_base_type> uv_grid[nx*ny*pols];
    //<first_row>ch1:pol1...pol4,ch2:pol1...pol4,<second_row>ch1:pol1...pol4,ch2:pol1...pol4
    complex<visibility_base_type> vis[] = {//row 1,ch 1
					   complex<visibility_base_type>(0,1),complex<visibility_base_type>(0,1),
					   complex<visibility_base_type>(0,1),complex<visibility_base_type>(0,1),
					   //row 1,ch 2
					   complex<visibility_base_type>(1,0),complex<visibility_base_type>(1,0),
					   complex<visibility_base_type>(1,0),complex<visibility_base_type>(1,0),
					   //row 2,ch 1
					   complex<visibility_base_type>(0,1),complex<visibility_base_type>(0,1),
					   complex<visibility_base_type>(0,1),complex<visibility_base_type>(0,1),
					   //row 2,ch 2
					   complex<visibility_base_type>(1,0),complex<visibility_base_type>(1,0),
					   complex<visibility_base_type>(1,0),complex<visibility_base_type>(1,0)};
    visibility_weights_base_type weights[] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
    unsigned int spw [] = {0,0};
    bool flags[] = {false,false,false,true,false,false,true,false,false,false,true,false,false,true,false,true};
    uvw_coord<uvw_base_type> uvw(1,2,3);
    polarization_gridding_policy_type polarization_policy(phase_transform,
							  (complex<grid_base_type>*)(uv_grid),
							  (complex<visibility_base_type>*)(vis),
							  (visibility_weights_base_type*)weights,
							  (bool*)flags,nx*ny,ch);
    for (size_t r = 0; r < rows; ++r){
	for (size_t c = 0; c < ch; ++c){
	  typename polarization_gridding_policy_type::trait_type::pol_vis_type vis;
	  polarization_policy.transform(r,spw[r],c,uvw,vis);
	  polarization_policy.grid_polarization_terms(25*nx+25,vis,1);
	}
    }
    REQUIRE(uvw._u == 1);
    REQUIRE(uvw._v == 2);
    REQUIRE(uvw._w == 3);
    //grid all the even numbers 0,2,4,6 (pol 0), weighted 1,3,5,7, but the last one is flagged
    REQUIRE(uv_grid[nx*ny*0 + 25*nx+25].real() == 18);
    REQUIRE(uv_grid[nx*ny*0 + 25*nx+25].imag() == 10);
    REQUIRE(uv_grid[nx*ny*1 + 25*nx+25].real() == 6);
    REQUIRE(uv_grid[nx*ny*1 + 25*nx+25].imag() == 12);
    REQUIRE(uv_grid[nx*ny*2 + 25*nx+25].real() == 15);
    REQUIRE(uv_grid[nx*ny*2 + 25*nx+25].imag() == 3);
    REQUIRE(uv_grid[nx*ny*3 + 25*nx+25].real() == 8);
    REQUIRE(uv_grid[nx*ny*3 + 25*nx+25].imag() == 12);
  }
  SECTION( "Testing the 4 correlation jones correcting policy" ){
    typedef float visibility_base_type;
    typedef float uvw_base_type;
    typedef float reference_wavelengths_base_type;
    typedef float convolution_base_type;
    typedef float visibility_weights_base_type;
    typedef float grid_base_type;
    
    typedef baseline_transform_policy<uvw_base_type, transform_disable_facet_rotation> baseline_transform_policy_type;
    typedef phase_transform_policy<visibility_base_type, uvw_base_type, transform_disable_phase_rotation> phase_transform_policy_type;
    typedef polarization_gridding_policy<visibility_base_type, uvw_base_type, visibility_weights_base_type, 
					 convolution_base_type, grid_base_type, phase_transform_policy_type, 
					 gridding_4_pol_enable_facet_based_jones_corrections> polarization_gridding_policy_type;
    baseline_transform_policy_type uvw_transform; //standard: no uvw rotation
    phase_transform_policy_type phase_transform; //standard: no phase rotation
    //emulate two rows in the measurement set
    size_t rows = 2;
    size_t ts = 2;
    size_t ant = 2;
    size_t dir = 2;
    size_t spw_count = 2;
    size_t ch = 2;
    size_t pols = 4;
    size_t nx = 50;
    size_t ny = 50;
    complex<grid_base_type> uv_grid[nx*ny*pols];
    complex<visibility_base_type> vis[] = {//row 1,ch 1
					   complex<visibility_base_type>(0,1),complex<visibility_base_type>(0,1),
					   complex<visibility_base_type>(0,1),complex<visibility_base_type>(0,1),
					   //row 1,ch 2
					   complex<visibility_base_type>(0,2),complex<visibility_base_type>(0,3),
					   complex<visibility_base_type>(0,4),complex<visibility_base_type>(0,5),
					   //row 2,ch 1
					   complex<visibility_base_type>(1,0),complex<visibility_base_type>(1,0),
					   complex<visibility_base_type>(1,0),complex<visibility_base_type>(1,0),
					   //row 2,ch 2
					   complex<visibility_base_type>(-1,0),complex<visibility_base_type>(-2,0),
					   complex<visibility_base_type>(-3,0),complex<visibility_base_type>(-4,0)
    };
					   
    visibility_weights_base_type weights[] = {//row 1,ch 1
					      1,2,3,4,
					      //row 1,ch 2
					      -1,-2,-3,-4,
					      //row 2,ch 1
					      5,6,7,8,
					      //row 2,ch 2
					      -5,-6,-7,-8
    };
    bool flags[] = {//row 1,ch 1
		    true,true,true,true,
		    //row 1,ch 2
		    false,false,false,false,
		    //row 2,ch 1
		    false,false,false,false,
		    //row 2,ch 2
		    false,false,false,false
    };
    //define jones terms for all time x antennae x directions x spws x channels:
    jones_2x2<visibility_base_type> jones_terms[] = {//time 1,antenna 1,dir 1,spw 1,ch 1
						     complex<visibility_base_type>(260,0),complex<visibility_base_type>(0,0),
						     complex<visibility_base_type>(0,0),complex<visibility_base_type>(260,0),
						     //time 1,antenna 1,dir 1,spw 1,ch 2
						     complex<visibility_base_type>(2,0),complex<visibility_base_type>(0,0),
						     complex<visibility_base_type>(0,0),complex<visibility_base_type>(2,0),
						     //time 1,antenna 1,dir 1,spw 2,ch 1
						     complex<visibility_base_type>(3,0),complex<visibility_base_type>(0,0),
						     complex<visibility_base_type>(0,0),complex<visibility_base_type>(3,0),
						     //time 1,antenna 1,dir 1,spw 2,ch 2
						     complex<visibility_base_type>(4,0),complex<visibility_base_type>(0,0),
						     complex<visibility_base_type>(0,0),complex<visibility_base_type>(4,0),
						     //time 1,antenna 1,dir 2,spw 1,ch 1
						     complex<visibility_base_type>(5,0),complex<visibility_base_type>(0,0),
						     complex<visibility_base_type>(0,0),complex<visibility_base_type>(5,0),
						     //time 1,antenna 1,dir 2,spw 1,ch 2
						     complex<visibility_base_type>(6,0),complex<visibility_base_type>(0,0),
						     complex<visibility_base_type>(0,0),complex<visibility_base_type>(6,0),
						     //time 1,antenna 1,dir 2,spw 2,ch 1
						     complex<visibility_base_type>(7,0),complex<visibility_base_type>(0,0),
						     complex<visibility_base_type>(0,0),complex<visibility_base_type>(7,0),
						     //time 1,antenna 1,dir 2,spw 2,ch 2
						     complex<visibility_base_type>(8,0),complex<visibility_base_type>(0,0),
						     complex<visibility_base_type>(0,0),complex<visibility_base_type>(8,0),
						     //time 1,antenna 2,dir 1,spw 1,ch 1
						     complex<visibility_base_type>(0,9),complex<visibility_base_type>(0,0),
						     complex<visibility_base_type>(0,0),complex<visibility_base_type>(0,9),
						     //time 1,antenna 2,dir 1,spw 1,ch 2
						     complex<visibility_base_type>(10,0),complex<visibility_base_type>(0,0),
						     complex<visibility_base_type>(0,0),complex<visibility_base_type>(10,0),
						     //time 1,antenna 2,dir 1,spw 2,ch 1
						     complex<visibility_base_type>(11,0),complex<visibility_base_type>(0,0),
						     complex<visibility_base_type>(0,0),complex<visibility_base_type>(11,0),
						     //time 1,antenna 2,dir 1,spw 2,ch 2
						     complex<visibility_base_type>(12,0),complex<visibility_base_type>(0,0),
						     complex<visibility_base_type>(0,0),complex<visibility_base_type>(12,0),
						     //time 1,antenna 2,dir 2,spw 1,ch 1
						     complex<visibility_base_type>(13,0),complex<visibility_base_type>(0,0),
						     complex<visibility_base_type>(0,0),complex<visibility_base_type>(13,0),
						     //time 1,antenna 2,dir 2,spw 1,ch 2
						     complex<visibility_base_type>(14,0),complex<visibility_base_type>(0,0),
						     complex<visibility_base_type>(0,0),complex<visibility_base_type>(14,0),
						     //time 1,antenna 2,dir 2,spw 2,ch 1
						     complex<visibility_base_type>(15,0),complex<visibility_base_type>(0,0),
						     complex<visibility_base_type>(0,0),complex<visibility_base_type>(15,0),
						     //time 1,antenna 2,dir 2,spw 2,ch 2
						     complex<visibility_base_type>(16,0),complex<visibility_base_type>(0,0),
						     complex<visibility_base_type>(0,0),complex<visibility_base_type>(16,0),
						     //time 2,antenna 1,dir 1,spw 1,ch 1
						     complex<visibility_base_type>(17,0),complex<visibility_base_type>(0,0),
						     complex<visibility_base_type>(0,0),complex<visibility_base_type>(17,0),
						     //time 2,antenna 1,dir 1,spw 1,ch 2
						     complex<visibility_base_type>(18,2),complex<visibility_base_type>(0,0),
						     complex<visibility_base_type>(0,0),complex<visibility_base_type>(2,-18),
						     //time 2,antenna 1,dir 1,spw 2,ch 1
						     complex<visibility_base_type>(19,0),complex<visibility_base_type>(0,0),
						     complex<visibility_base_type>(0,0),complex<visibility_base_type>(19,0),
						     //time 2,antenna 1,dir 1,spw 2,ch 2
						     complex<visibility_base_type>(20,0),complex<visibility_base_type>(0,0),
						     complex<visibility_base_type>(0,0),complex<visibility_base_type>(20,0),
						     //time 2,antenna 1,dir 2,spw 1,ch 1
						     complex<visibility_base_type>(21,0),complex<visibility_base_type>(0,0),
						     complex<visibility_base_type>(0,0),complex<visibility_base_type>(21,0),
						     //time 2,antenna 1,dir 2,spw 1,ch 2
						     complex<visibility_base_type>(22,0),complex<visibility_base_type>(0,0),
						     complex<visibility_base_type>(0,0),complex<visibility_base_type>(22,0),
						     //time 2,antenna 1,dir 2,spw 2,ch 1
						     complex<visibility_base_type>(23,0),complex<visibility_base_type>(0,0),
						     complex<visibility_base_type>(0,0),complex<visibility_base_type>(23,0),
						     //time 2,antenna 1,dir 2,spw 2,ch 2
						     complex<visibility_base_type>(24,0),complex<visibility_base_type>(0,0),
						     complex<visibility_base_type>(0,0),complex<visibility_base_type>(24,0),
						     //time 2,antenna 2,dir 1,spw 1,ch 1
						     complex<visibility_base_type>(25,0),complex<visibility_base_type>(0,0),
						     complex<visibility_base_type>(0,0),complex<visibility_base_type>(25,0),
						     //time 2,antenna 2,dir 1,spw 1,ch 2
						     complex<visibility_base_type>(26,0),complex<visibility_base_type>(0,0),
						     complex<visibility_base_type>(0,0),complex<visibility_base_type>(0,26),
						     //time 2,antenna 2,dir 1,spw 2,ch 1
						     complex<visibility_base_type>(27,0),complex<visibility_base_type>(0,0),
						     complex<visibility_base_type>(0,0),complex<visibility_base_type>(27,0),
						     //time 2,antenna 2,dir 1,spw 2,ch 2
						     complex<visibility_base_type>(28,0),complex<visibility_base_type>(0,0),
						     complex<visibility_base_type>(0,0),complex<visibility_base_type>(28,0),
						     //time 2,antenna 2,dir 2,spw 1,ch 1
						     complex<visibility_base_type>(29,0),complex<visibility_base_type>(0,0),
						     complex<visibility_base_type>(0,0),complex<visibility_base_type>(29,0),
						     //time 2,antenna 2,dir 2,spw 1,ch 2
						     complex<visibility_base_type>(30,0),complex<visibility_base_type>(0,0),
						     complex<visibility_base_type>(0,0),complex<visibility_base_type>(30,0),
						     //time 2,antenna 2,dir 2,spw 2,ch 1
						     complex<visibility_base_type>(31,0),complex<visibility_base_type>(0,0),
						     complex<visibility_base_type>(0,0),complex<visibility_base_type>(31,0),
						     //time 2,antenna 2,dir 2,spw 2,ch 2
						     complex<visibility_base_type>(32,0),complex<visibility_base_type>(0,0),
						     complex<visibility_base_type>(0,0),complex<visibility_base_type>(32,0)
    };
    unsigned int antenna_1 [] = {0,0};
    unsigned int antenna_2 [] = {1,1};
    size_t time[] = {0,1};
    unsigned int spw [] = {0,0}; //visibilties of only the first spw
    uvw_coord<uvw_base_type> uvw(1,2,3);
    polarization_gridding_policy_type polarization_policy(phase_transform,
							  (complex<grid_base_type>*)(uv_grid),
							  (complex<visibility_base_type>*)(vis),
							  (visibility_weights_base_type*)weights,
							  (bool*)flags,
							  nx*ny,ch,
							  (jones_2x2<visibility_base_type>*)jones_terms,
							  (unsigned int*)antenna_1,
							  (unsigned int*)antenna_2,
							  (size_t*)time,
							  ant,
							  0, //direction index
							  dir,
							  ts,
							  spw_count
 							);
    for (size_t r = 0; r < rows; ++r){
	for (size_t c = 0; c < ch; ++c){
	  typename polarization_gridding_policy_type::trait_type::pol_vis_type vis;
	  polarization_policy.transform(r,spw[r],c,uvw,vis);
	  polarization_policy.grid_polarization_terms(25*nx+25,vis,1);
	}
    }
    REQUIRE(uvw._u == 1);
    REQUIRE(uvw._v == 2);
    REQUIRE(uvw._w == 3);
    
    REQUIRE(uv_grid[nx*ny*0 + 25*nx+25].real() == 4465);
    REQUIRE(uv_grid[nx*ny*0 + 25*nx+25].imag() == 220);
    REQUIRE(uv_grid[nx*ny*1 + 25*nx+25].real() == 3174);
    REQUIRE(uv_grid[nx*ny*1 + 25*nx+25].imag() == -5736);
    REQUIRE(uv_grid[nx*ny*2 + 25*nx+25].real() == 4067);
    REQUIRE(uv_grid[nx*ny*2 + 25*nx+25].imag() == -10068);
    REQUIRE(uv_grid[nx*ny*3 + 25*nx+25].real() == -11576);
    REQUIRE(uv_grid[nx*ny*3 + 25*nx+25].imag() == -2064);
  }  
}
TEST_CASE( "TESTING THE CONVOLUTION POLICIES" )
{
  typedef float visibility_base_type;
  typedef float uvw_base_type;
  typedef float reference_wavelengths_base_type;
  typedef float convolution_base_type;
  typedef float visibility_weights_base_type;
  typedef float grid_base_type;
  
  class pol_gridding_placeholder {
  public:
    std::complex<grid_base_type> * _output_grids;
    pol_gridding_placeholder(std::complex<grid_base_type> * output_grids): _output_grids(output_grids) {}
    inline void grid_polarization_terms(std::size_t term_flat_index, convolution_base_type convolution_weight) __restrict__ {
	std::complex<visibility_base_type> vis(1,1);
	_output_grids[term_flat_index] += convolution_weight*vis;
      }
      inline void grid_polarization_conjugate_terms(std::size_t term_flat_index, convolution_base_type convolution_weight) __restrict__ {
	std::complex<visibility_base_type> conj(1,-2);
	_output_grids[term_flat_index] += convolution_weight*conj;
      }
  };
//   SECTION( "TESTING THE PRECOMPUTED CONVOLUTION POLICY" )
//   {
//     std::complex<grid_base_type> * grid = new std::complex<grid_base_type>[512*512]();
//     typedef imaging::convolution_policy<convolution_base_type,uvw_base_type,grid_base_type,
// 					pol_gridding_placeholder,convolution_precomputed_fir> convolution_policy_type;
//     pol_gridding_placeholder polarization_policy(grid);
//     convolution_base_type conv [] = {1,2,3,4,5,6,
// 				     7,8,9,10,11,12,
// 				     13,14,15,16,17,18,
// 				     19,20,21,22,23,24,
// 				     25,26,27,28,29,30,
// 				     31,32,33,34,35,36};
//     convolution_policy_type convolution_policy(512,512,
// 					       2,3,(convolution_base_type*) conv, polarization_policy);
//     {
//       uvw_coord<uvw_base_type> uvw(0,0);
//       convolution_policy.convolve(uvw, &pol_gridding_placeholder::grid_polarization_terms);
//       convolution_policy.convolve(uvw, &pol_gridding_placeholder::grid_polarization_conjugate_terms);
//       // starting from 1 cell to the left do 3 additions per dimension, over two cells per direction
//       REQUIRE(is_close(grid[512*int(256-1)+int(256-1)].real(),(1+2+3+7+8+9+13+14+15)*2));
//       REQUIRE(is_close(grid[512*int(256-1)+int(256)].real(),(4+5+6+10+11+12+16+17+18)*2));
//       REQUIRE(is_close(grid[512*int(256)+int(256-1)].real(),(19+20+21+25+26+27+31+32+33)*2));
//       REQUIRE(is_close(grid[512*int(256)+int(256)].real(),(22+23+24+28+29+30+34+35+36)*2));
//     
//       REQUIRE(is_close(grid[512*int(256-1)+int(256-1)].imag(),(1+2+3+7+8+9+13+14+15)*-1));
//       REQUIRE(is_close(grid[512*int(256-1)+int(256)].imag(),(4+5+6+10+11+12+16+17+18)*-1));
//       REQUIRE(is_close(grid[512*int(256)+int(256-1)].imag(),(19+20+21+25+26+27+31+32+33)*-1));
//       REQUIRE(is_close(grid[512*int(256)+int(256)].imag(),(22+23+24+28+29+30+34+35+36)*-1));
//     }
//     {
//       uvw_coord<uvw_base_type> uvw(256,256);
//       convolution_policy.convolve(uvw, &pol_gridding_placeholder::grid_polarization_terms);
//       convolution_policy.convolve(uvw, &pol_gridding_placeholder::grid_polarization_conjugate_terms);
//       //only the top 3x3 cells of the convolution function should be on the grid
//       REQUIRE(is_close(grid[512*int(512-1)+int(512-1)].real(),(1+2+3+7+8+9+13+14+15)*2));   
//       REQUIRE(is_close(grid[512*int(512-1)+int(512-1)].imag(),(1+2+3+7+8+9+13+14+15)*-1));   
//     }
//     {
//       uvw_coord<uvw_base_type> uvw(-256,-256);
//       convolution_policy.convolve(uvw, &pol_gridding_placeholder::grid_polarization_terms);
//       convolution_policy.convolve(uvw, &pol_gridding_placeholder::grid_polarization_conjugate_terms);
//       //only the bottom 3x3 cells of the convolution function should be on the grid
//       REQUIRE(grid[512*int(0)+int(0)].real() == (22+23+24+28+29+30+34+35+36)*2);   
//       REQUIRE(grid[512*int(0)+int(0)].imag() == (22+23+24+28+29+30+34+35+36)*-1);   
//     }
//     delete[] grid;
//   }
//   SECTION( "TESTING THE PRECOMPUTED CONVOLUTION POLICY AND SAMPLE FUNCTION GRIDDING" )
//   {
//     std::complex<grid_base_type> * grid = new std::complex<grid_base_type>[512*512]();
//     std::complex<grid_base_type> * samp = new std::complex<grid_base_type>[512*512]();
//     typedef imaging::convolution_policy<convolution_base_type,uvw_base_type,grid_base_type,
// 					pol_gridding_placeholder,convolution_with_sampling_function_gridding_using_precomputed_fir> convolution_policy_type;
//     pol_gridding_placeholder polarization_policy(grid);
//     convolution_base_type conv [] = {1,2,3,4,5,6,
// 				     7,8,9,10,11,12,
// 				     13,14,15,16,17,18,
// 				     19,20,21,22,23,24,
// 				     25,26,27,28,29,30,
// 				     31,32,33,34,35,36};
//     convolution_policy_type convolution_policy(512,512,
// 					       2,3,(convolution_base_type*) conv, polarization_policy, samp);
//     {
//       uvw_coord<uvw_base_type> uvw(0,0);
//       convolution_policy.convolve(uvw, &pol_gridding_placeholder::grid_polarization_terms);
//       convolution_policy.convolve(uvw, &pol_gridding_placeholder::grid_polarization_conjugate_terms);
//       // starting from 1 cell to the left do 3 additions per dimension, over two cells per direction
//       REQUIRE(grid[512*int(256-1)+int(256-1)].real()==(1+2+3+7+8+9+13+14+15)*2);
//       REQUIRE(grid[512*int(256-1)+int(256)].real()==(4+5+6+10+11+12+16+17+18)*2);
//       REQUIRE(grid[512*int(256)+int(256-1)].real()==(19+20+21+25+26+27+31+32+33)*2);
//       REQUIRE(grid[512*int(256)+int(256)].real()==(22+23+24+28+29+30+34+35+36)*2);
//     
//       REQUIRE(grid[512*int(256-1)+int(256-1)].imag()==(1+2+3+7+8+9+13+14+15)*-1);
//       REQUIRE(grid[512*int(256-1)+int(256)].imag()==(4+5+6+10+11+12+16+17+18)*-1);
//       REQUIRE(grid[512*int(256)+int(256-1)].imag()==(19+20+21+25+26+27+31+32+33)*-1);
//       REQUIRE(grid[512*int(256)+int(256)].imag()==(22+23+24+28+29+30+34+35+36)*-1);
//       
//       REQUIRE(samp[512*int(256-1)+int(256-1)].real()==(1+2+3+7+8+9+13+14+15)*2);
//       REQUIRE(samp[512*int(256-1)+int(256)].real()==(4+5+6+10+11+12+16+17+18)*2);
//       REQUIRE(samp[512*int(256)+int(256-1)].real()==(19+20+21+25+26+27+31+32+33)*2);
//       REQUIRE(samp[512*int(256)+int(256)].real()==(22+23+24+28+29+30+34+35+36)*2);
//       
//       REQUIRE(samp[512*int(256-1)+int(256-1)].imag()==0);
//       REQUIRE(samp[512*int(256-1)+int(256)].imag()==0);
//       REQUIRE(samp[512*int(256)+int(256-1)].imag()==0);
//       REQUIRE(samp[512*int(256)+int(256)].imag()==0);
//     }
//     {
//       uvw_coord<uvw_base_type> uvw(256,256);
//       convolution_policy.convolve(uvw, &pol_gridding_placeholder::grid_polarization_terms);
//       convolution_policy.convolve(uvw, &pol_gridding_placeholder::grid_polarization_conjugate_terms);
//       //only the top 3x3 cells of the convolution function should be on the grid
//       REQUIRE(is_close(grid[512*int(512-1)+int(512-1)].real(),(1+2+3+7+8+9+13+14+15)*2));   
//       REQUIRE(is_close(grid[512*int(512-1)+int(512-1)].imag(),(1+2+3+7+8+9+13+14+15)*-1));   
//       REQUIRE(is_close(samp[512*int(512-1)+int(512-1)].real(),(1+2+3+7+8+9+13+14+15)*2));   
//       REQUIRE(is_close(samp[512*int(512-1)+int(512-1)].imag(),0));
//     }
//     {
//       uvw_coord<uvw_base_type> uvw(-256,-256);
//       convolution_policy.convolve(uvw, &pol_gridding_placeholder::grid_polarization_terms);
//       convolution_policy.convolve(uvw, &pol_gridding_placeholder::grid_polarization_conjugate_terms);
//       //only the bottom 3x3 cells of the convolution function should be on the grid
//       REQUIRE(is_close(grid[512*int(0)+int(0)].real(),(22+23+24+28+29+30+34+35+36)*2));   
//       REQUIRE(is_close(grid[512*int(0)+int(0)].imag(),(22+23+24+28+29+30+34+35+36)*-1));   
//       REQUIRE(is_close(samp[512*int(0)+int(0)].real(),(22+23+24+28+29+30+34+35+36)*2));   
//       REQUIRE(is_close(samp[512*int(0)+int(0)].imag(),0));
//     }
//     delete[] grid;
//   }
}