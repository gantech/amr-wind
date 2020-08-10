#ifndef DERIVE_STENCILS_H_
#define DERIVE_STENCILS_H_

#include "amr-wind/core/Field.H"
#include <AMReX_FArrayBox.H>
#include <cmath>

namespace amr_wind {

struct StencilInterior
{
    // First derivatives
    // fx= (ax*f_{i+1,j,k}+bx*f_{i,j,k}+cx*f_{i-1,j,k})/dx
    // fy= (ay*f_{i,j+1,k}+by*f_{i,j,k}+cy*f_{i,j-1,k})/dy
    // fz= (az*f_{i,j,k+1}+bz*f_{i,j,k}+cz*f_{i,j,k-1})/dz
    static constexpr amrex::Real c00 =  0.5; // ax 
    static constexpr amrex::Real c01 =  0.0; // bx
    static constexpr amrex::Real c02 = -0.5; // cx
    static constexpr amrex::Real c10 =  c00; // ay
    static constexpr amrex::Real c11 =  c01; // by
    static constexpr amrex::Real c12 =  c02; // cy
    static constexpr amrex::Real c20 =  c00; // az
    static constexpr amrex::Real c21 =  c01; // bz
    static constexpr amrex::Real c22 =  c02; // cz

    // Second derivatives
    // fxx= (axx*f_{i+1,j,k}+bxx*f_{i,j,k}+cxx*f_{i-1,j,k})/dx^2
    // fyy= (ayy*f_{i,j+1,k}+byy*f_{i,j,k}+cyy*f_{i,j-1,k})/dy^2
    // fzz= (azz*f_{i,j,k+1}+bzz*f_{i,j,k}+czz*f_{i,j,k-1})/dz^2
    static constexpr amrex::Real s00 =  1.0; // axx 
    static constexpr amrex::Real s01 = -2.0; // bxx
    static constexpr amrex::Real s02 =  1.0; // cxx
    static constexpr amrex::Real s10 =  s00; // ayy
    static constexpr amrex::Real s11 =  s01; // byy
    static constexpr amrex::Real s12 =  s02; // cyy
    static constexpr amrex::Real s20 =  s00; // azz
    static constexpr amrex::Real s21 =  s01; // bzz
    static constexpr amrex::Real s22 =  s02; // czz

};

struct StencilILO
{
    static constexpr amrex::Real c00 =  1./3.;
    static constexpr amrex::Real c01 =  1.;
    static constexpr amrex::Real c02 = -4./3.;
    static constexpr amrex::Real c10 =  0.5;
    static constexpr amrex::Real c11 =  0.0;
    static constexpr amrex::Real c12 = -0.5;
    static constexpr amrex::Real c20 =  c10;
    static constexpr amrex::Real c21 =  c11;
    static constexpr amrex::Real c22 =  c12;
    
    static constexpr amrex::Real s00 =  4./3.;  // axx 
    static constexpr amrex::Real s01 = -4;      // bxx
    static constexpr amrex::Real s02 =  8./3.;  // cxx
    static constexpr amrex::Real s10 =  1;      // ayy
    static constexpr amrex::Real s11 =  -2;     // byy
    static constexpr amrex::Real s12 =  1;      // cyy
    static constexpr amrex::Real s20 =  s10;    // azz
    static constexpr amrex::Real s21 =  s11;    // bzz
    static constexpr amrex::Real s22 =  s12;    // czz

};

struct StencilJLO
{
    static constexpr amrex::Real c00 =  0.5;
    static constexpr amrex::Real c01 =  0.0;
    static constexpr amrex::Real c02 = -0.5;
    static constexpr amrex::Real c10 =  1./3;
    static constexpr amrex::Real c11 =  1;
    static constexpr amrex::Real c12 = -4./3.;
    static constexpr amrex::Real c20 =  c00;
    static constexpr amrex::Real c21 =  c01;
    static constexpr amrex::Real c22 =  c02;

    static constexpr amrex::Real s00 =  1.0;    // axx 
    static constexpr amrex::Real s01 = -2.0;    // bxx
    static constexpr amrex::Real s02 =  1.0;    // cxx
    static constexpr amrex::Real s10 =  4./3.;  // ayy
    static constexpr amrex::Real s11 =  -4.;    // byy
    static constexpr amrex::Real s12 =  8./3.;  // cyy
    static constexpr amrex::Real s20 =  s00;    // azz
    static constexpr amrex::Real s21 =  s01;    // bzz
    static constexpr amrex::Real s22 =  s02;    // czz

};

struct StencilKLO
{
    static constexpr amrex::Real c00 =  0.5;
    static constexpr amrex::Real c01 =  0.0;
    static constexpr amrex::Real c02 = -0.5;
    static constexpr amrex::Real c10 =  c00;
    static constexpr amrex::Real c11 =  c01;
    static constexpr amrex::Real c12 =  c02;
    static constexpr amrex::Real c20 =  1./3.;
    static constexpr amrex::Real c21 =  1.;
    static constexpr amrex::Real c22 = -4./3.;
    
    static constexpr amrex::Real s00 =  1.0;    // axx 
    static constexpr amrex::Real s01 = -2.0;    // bxx
    static constexpr amrex::Real s02 =  1.0;    // cxx
    static constexpr amrex::Real s10 =  s00;    // ayy
    static constexpr amrex::Real s11 =  s01;    // byy
    static constexpr amrex::Real s12 =  s02;    // cyy
    static constexpr amrex::Real s20 =  4./3.;  // azz
    static constexpr amrex::Real s21 =  -4.;    // bzz
    static constexpr amrex::Real s22 =  8./3.;  // czz
     
};

struct StencilIHI
{
    static constexpr amrex::Real c00 =  4./3.;
    static constexpr amrex::Real c01 =  -1;
    static constexpr amrex::Real c02 =  -1./3.;
    static constexpr amrex::Real c10 =  0.5;
    static constexpr amrex::Real c11 =  0.0;
    static constexpr amrex::Real c12 = -0.5;
    static constexpr amrex::Real c20 =  c10;
    static constexpr amrex::Real c21 =  c11;
    static constexpr amrex::Real c22 =  c12;

    static constexpr amrex::Real s00 = 8./3.;   // axx 
    static constexpr amrex::Real s01 = -4.;     // bxx
    static constexpr amrex::Real s02 = 4./3.;   // cxx
    static constexpr amrex::Real s10 =  1.;     // ayy
    static constexpr amrex::Real s11 = -2.;     // byy
    static constexpr amrex::Real s12 =  1.;     // cyy
    static constexpr amrex::Real s20 =  s10;    // azz
    static constexpr amrex::Real s21 =  s11;    // bzz
    static constexpr amrex::Real s22 =  s12;    // czz

};

struct StencilJHI
{
    static constexpr amrex::Real c00 =  0.5;
    static constexpr amrex::Real c01 =  0.0;
    static constexpr amrex::Real c02 = -0.5;
    static constexpr amrex::Real c10 =  4./3.;
    static constexpr amrex::Real c11 = -1.;
    static constexpr amrex::Real c12 = -1./3.;
    static constexpr amrex::Real c20 =  c00;
    static constexpr amrex::Real c21 =  c01;
    static constexpr amrex::Real c22 =  c02;
    
    static constexpr amrex::Real s00 =  1.0;    // axx 
    static constexpr amrex::Real s01 = -2.0;    // bxx
    static constexpr amrex::Real s02 =  1.0;    // cxx
    static constexpr amrex::Real s10 =  8./3.;  // ayy
    static constexpr amrex::Real s11 =  -4.;    // byy
    static constexpr amrex::Real s12 =  4./3.;  // cyy
    static constexpr amrex::Real s20 =  s00;    // azz
    static constexpr amrex::Real s21 =  s01;    // bzz
    static constexpr amrex::Real s22 =  s02;    // czz
};

struct StencilKHI
{
    static constexpr amrex::Real c00 =  0.5;
    static constexpr amrex::Real c01 =  0.0;
    static constexpr amrex::Real c02 = -0.5;
    static constexpr amrex::Real c10 =  c00;
    static constexpr amrex::Real c11 =  c01;
    static constexpr amrex::Real c12 =  c02;
    static constexpr amrex::Real c20 =  4./3.;
    static constexpr amrex::Real c21 = -1.;
    static constexpr amrex::Real c22 = -1./3.;

    static constexpr amrex::Real s00 =  1.0;   // axx 
    static constexpr amrex::Real s01 = -2.0;   // bxx
    static constexpr amrex::Real s02 =  1.0;   // cxx
    static constexpr amrex::Real s10 =  s00;   // ayy
    static constexpr amrex::Real s11 =  s01;   // byy
    static constexpr amrex::Real s12 =  s02;   // cyy
    static constexpr amrex::Real s20 =  8./3.; // azz
    static constexpr amrex::Real s21 =  -4.;   // bzz
    static constexpr amrex::Real s22 =  4./3.; // czz

};


} // namespace amr_wind
#endif
