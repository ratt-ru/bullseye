#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
#include "catch.hpp"
#include "../phase_transform_policies.h"
#include <complex>
#include <cmath>
using namespace imaging;
using namespace std;

bool is_close(double val, double expect){
  #define EPSILON 1
  return abs(expect - val) <= EPSILON;
}

TEST_CASE( "Testing phase transform policies" ) {
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
    phase_transform_policy<double,double,transform_enable_phase_rotation_lefthanded_ra_dec> a_policy(126000,144000,
												     100000,135000);
    complex<double> visibility(3.0f,5.0f);
    uvw_coord<double> uvw(1.5f,6.0f,12.0f);
    complex<double> ans(-3.304262109,-7.285729333);
    a_policy.transform(visibility,uvw);
    REQUIRE(visibility.real() == ans.real());
    REQUIRE(visibility.imag() == ans.imag());
    REQUIRE(uvw._u == 1.5f);
    REQUIRE(uvw._v == 6.0f);
    REQUIRE(uvw._w == 12.0f);
  }
}