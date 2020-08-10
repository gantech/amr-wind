#ifndef DERIVE_K_H_
#define DERIVE_K_H_

#include "amr-wind/core/Field.H"
#include <AMReX_FArrayBox.H>
#include <cmath>
#include "derive_stencils.H"

namespace amr_wind {

template <typename Stencil>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void gradient(
    int i,
    int j,
    int k,
    amrex::Real idx,
    amrex::Real idy,
    amrex::Real idz,
    amrex::Array4<amrex::Real const> const& phi,
    amrex::Array4<amrex::Real> const& gradphi,
    int ncomp) noexcept
{
    using namespace amrex;

    amrex::Real cp1, c, cm1;
    for (int icomp=0; icomp < ncomp; icomp++) {
        cp1 = Stencil::c00;
        c   = Stencil::c01;
        cm1 = Stencil::c02;
        gradphi(i,j,k,icomp*AMREX_SPACEDIM+0) = (cp1*phi(i+1,j,k,icomp) + c*phi(i,j,k,icomp) + cm1*phi(i-1,j,k,icomp)) * idx;
        cp1 = Stencil::c10;
        c   = Stencil::c11;
        cm1 = Stencil::c12;
        gradphi(i,j,k,icomp*AMREX_SPACEDIM+1) = (cp1*phi(i,j+1,k,icomp) + c*phi(i,j,k,icomp) + cm1*phi(i,j-1,k,icomp)) * idy;
        cp1 = Stencil::c20;
        c   = Stencil::c21;
        cm1 = Stencil::c22;
        gradphi(i,j,k,icomp*AMREX_SPACEDIM+2) = (cp1*phi(i,j,k+1,icomp) + c*phi(i,j,k,icomp) + cm1*phi(i,j,k-1,icomp)) * idz;
    }

}

template <typename Stencil>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void laplacian(
    int i,
    int j,
    int k,
    amrex::Real idx,
    amrex::Real idy,
    amrex::Real idz,
    amrex::Array4<amrex::Real const> const& phi,
    amrex::Array4<amrex::Real> const& Laplacephi,
    int ncomp) noexcept
{

    amrex::Real sp1, s, sm1, d2phidx2, d2phidy2,d2phidz2;
    for (int icomp=0; icomp < ncomp; icomp++) {
        sp1 = Stencil::s00;
        s   = Stencil::s01;
        sm1 = Stencil::s02;
        d2phidx2 = (sp1*phi(i+1,j,k,icomp) + s*phi(i,j,k,icomp) + sm1*phi(i-1,j,k,icomp))*idx*idx;
        sp1 = Stencil::s10;
        s   = Stencil::s11;
        sm1 = Stencil::s12;
        d2phidy2 = (sp1*phi(i,j+1,k,icomp) + s*phi(i,j,k,icomp) + sm1*phi(i,j-1,k,icomp))*idy*idy;
        sp1 = Stencil::s20;
        s   = Stencil::s21;
        sm1 = Stencil::s22;
        d2phidz2 = (sp1*phi(i,j,k+1,icomp) + s*phi(i,j,k,icomp) + sm1*phi(i,j,k-1,icomp))*idz*idz;
        
        Laplacephi(i,j,k,icomp) = d2phidx2+d2phidy2+d2phidz2; 
    }
}


template <typename Stencil>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real curvature(
    int i,
    int j,
    int k,
    amrex::Real idx,
    amrex::Real idy,
    amrex::Real idz,
    amrex::Array4<amrex::Real const> const& gradphi) noexcept
{

    amrex::Real cp1, c, cm1;

    cp1 = Stencil::c00;
    c   = Stencil::c01; 
    cm1 = Stencil::c02;
    const amrex::Real phix  = gradphi(i,j,k,0);
    const amrex::Real phixx = (cp1*gradphi(i+1,j,k,0) + c*gradphi(i,j,k,0) + cm1*gradphi(i-1,j,k,0)) * idx;
    
    cp1 = Stencil::c10;
    c   = Stencil::c11;
    cm1 = Stencil::c12;
    const amrex::Real phiy  = gradphi(i,j,k,1);
    const amrex::Real phixy = (cp1*gradphi(i,j+1,k,0) + c*gradphi(i,j,k,0) + cm1*gradphi(i,j-1,k,0))*idy;
    const amrex::Real phiyy = (cp1*gradphi(i,j+1,k,1) + c*gradphi(i,j,k,1) + cm1*gradphi(i,j-1,k,1))*idy;
    
    cp1 = Stencil::c20;
    c   = Stencil::c21;
    cm1 = Stencil::c22;
    const amrex::Real phiz  = gradphi(i,j,k,2);
    const amrex::Real phixz = (cp1*gradphi(i,j,k+1,0) + c*gradphi(i,j,k,0) + cm1*gradphi(i,j,k-1,0)) *idz;
    const amrex::Real phiyz = (cp1*gradphi(i,j,k+1,1) + c*gradphi(i,j,k,1) + cm1*gradphi(i,j,k-1,1)) *idz;
    const amrex::Real phizz = (cp1*gradphi(i,j,k+1,2) + c*gradphi(i,j,k,2) + cm1*gradphi(i,j,k-1,2)) *idz;
    
    amrex::Real curv= -(phix*phix*phiyy - 2.*phix*phiy*phixy + phiy*phiy*phixx + phix*phix*phizz
                -2.*phix*phiz*phixz + phiz*phiz*phixx + phiy*phiy*phizz - 2*phiy*phiz*phiyz 
                + phiz*phiz*phiyy)/std::pow(phix*phix+phiy*phiy+phiz*phiz,1.5);
       
    return curv;
}

template <typename Stencil>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real strainrate(
    int i,
    int j,
    int k,
    amrex::Real idx,
    amrex::Real idy,
    amrex::Real idz,
    amrex::Array4<amrex::Real const> const& vel) noexcept
{

    amrex::Real cp1, c, cm1;

    cp1 = Stencil::c00;
    c   = Stencil::c01;
    cm1 = Stencil::c02;

    const amrex::Real ux = (cp1*vel(i+1,j,k,0) + c*vel(i,j,k,0) + cm1*vel(i-1,j,k,0)) * idx;
    const amrex::Real vx = (cp1*vel(i+1,j,k,1) + c*vel(i,j,k,1) + cm1*vel(i-1,j,k,1)) * idx;
    const amrex::Real wx = (cp1*vel(i+1,j,k,2) + c*vel(i,j,k,2) + cm1*vel(i-1,j,k,2)) * idx;

    cp1 = Stencil::c10;
    c   = Stencil::c11;
    cm1 = Stencil::c12;

    const amrex::Real uy = (cp1*vel(i,j+1,k,0) + c*vel(i,j,k,0) + cm1*vel(i,j-1,k,0)) * idy;
    const amrex::Real vy = (cp1*vel(i,j+1,k,1) + c*vel(i,j,k,1) + cm1*vel(i,j-1,k,1)) * idy;
    const amrex::Real wy = (cp1*vel(i,j+1,k,2) + c*vel(i,j,k,2) + cm1*vel(i,j-1,k,2)) * idy;

    cp1 = Stencil::c20;
    c   = Stencil::c21;
    cm1 = Stencil::c22;

    const amrex::Real uz = (cp1*vel(i,j,k+1,0) + c*vel(i,j,k,0) + cm1*vel(i,j,k-1,0)) * idz;
    const amrex::Real vz = (cp1*vel(i,j,k+1,1) + c*vel(i,j,k,1) + cm1*vel(i,j,k-1,1)) * idz;
    const amrex::Real wz = (cp1*vel(i,j,k+1,2) + c*vel(i,j,k,2) + cm1*vel(i,j,k-1,2)) * idz;

    return std::sqrt(2.0 * ux*ux + 2.0 * vy*vy + 2.0 * wz*wz
                     + (uy+vx)*(uy+vx) + (vz+wy)*(vz+wy) + (wx+uz)*(wx+uz));
}

/** Compute strain-rate for a given field
 *  \ingroup field_ops
 *
 *  \tparam FType Field or ScratchField
 *  \param field [out] The computed magnitude of strain-rate
 *  \param field [in] The velocity field
 */
template<typename FType>
void compute_strainrate(FType& field, const Field& velocity);

/** Compute gradient of a scalar or vector field
 *  \ingroup field_ops
 *
 *  \f$\nabla \mathbf{f}\f$
 *
 *  \tparam FType Field or ScratchField
 *  \param gradf [out] The gradient of the input field
 *  \param field [in] Field on which gradient is to be computed
 */
template<typename FType>
void compute_gradient(FType& gradf, const Field& field);

/** Compute laplacian of a scalar or vector field
 *  \ingroup field_ops
 *
 *  \f$\nabla^2 \mathbf{f}\f$
 *
 *  \tparam FType Field or ScratchField
 *  \param gradf [out] The gradient of the input field
 *  \param field [in] Field on which gradient is to be computed
 */
template<typename FType>
void compute_laplacian(FType& laplacef, const Field& field);

template<typename FType>
void compute_curvature(FType& curvf, const Field& field);

/** Normalize a given vector field
 *  \ingroup field_ops
 *
 *  \f$ \mathbf{f} = \mathbf{f} / |\mathbf{f}|\f$
 *  \tparam FType Field or ScratchField
 *  \param field [inout] Field to be normalized
 */
template<typename FType>
void normalize_field(FType& field);

} // namespace amr_wind
#endif
