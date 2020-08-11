#ifndef KSGS_KC2000_UTILS_H
#define KSGS_KC2000_UTILS_H

#include "amr-wind/fvm/fvm_utils.H"
#include "AMReX_Array4.H"
#include "AMReX_Geometry.H"
#include "AMReX_MFIter.H"

namespace amr_wind {
namespace turbulence {

template<typename FTypeOut>
struct KC2000NonlinStress
{
    KC2000NonlinStress(FTypeOut& mij_nonlin, const Field& vel, amrex::Real cs, amrex::Real c1, amrex::Real c2)
        : m_mij_nonlin(mij_nonlin), m_vel(vel),
          m_cs(cs), m_c1(c1), m_c2(c2)
    {
        AMREX_ALWAYS_ASSERT(
            mij_nonlin.num_comp() == AMREX_SPACEDIM * AMREX_SPACEDIM);
    }

    template <typename Stencil>
    void apply(const int lev, const amrex::MFIter& mfi) const
    {
        const auto& geom = m_vel.repo().mesh().Geom(lev);
        const auto& idx = geom.InvCellSizeArray();
        const auto& dx = geom.CellSizeArray();
        const auto& mij_nonlin_arr = m_mij_nonlin(lev).array(mfi);
        const auto& vel_arr = m_vel(lev).const_array(mfi);
        const amrex::Real cs = this->m_cs;
        const amrex::Real c1 = this->m_c1;
        const amrex::Real c2 = this->m_c2;
        
        const auto& bx_in = mfi.tilebox();
        const auto& bx = Stencil::box(bx_in, geom);
        if (bx.isEmpty()) return;

        amrex::ParallelFor(
            bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                amrex::Real gv[AMREX_SPACEDIM*AMREX_SPACEDIM];
                amrex::Real cp1, c, cm1;
                for (int icomp = 0; icomp < AMREX_SPACEDIM; icomp++) {
                    cp1 = Stencil::c00;
                    c = Stencil::c01;
                    cm1 = Stencil::c02;
                    gv[icomp*AMREX_SPACEDIM+0] = (cp1*vel_arr(i+1,j,k,icomp) + c*vel_arr(i,j,k,icomp) + cm1*vel_arr(i-1,j,k,icomp)) * idx[0];
                    
                    cp1 = Stencil::c10;
                    c   = Stencil::c11;
                    cm1 = Stencil::c12;
                    gv[icomp*AMREX_SPACEDIM+1] = (cp1*vel_arr(i,j+1,k,icomp) + c*vel_arr(i,j,k,icomp) + cm1*vel_arr(i,j-1,k,icomp)) * idx[1];
                
                    cp1 = Stencil::c20;
                    c   = Stencil::c21;
                    cm1 = Stencil::c22;
                    gv[icomp*AMREX_SPACEDIM+2] = (cp1*vel_arr(i,j,k+1,icomp) + c*vel_arr(i,j,k,icomp) + cm1*vel_arr(i,j,k-1,icomp)) * idx[2];
                }
                                
                const amrex::Real sijsij =
                    std::sqrt(gv[0]*gv[0] + gv[5]*gv[5] + gv[8]*gv[8]
                              + 0.5 * ((gv[1]+gv[4])*(gv[1]+gv[4])
                                       + (gv[6]+gv[7])*(gv[6]+gv[7])
                                       + (gv[6]+gv[2])*(gv[6]+gv[2])) );

                const amrex::Real ds = std::cbrt(dx[0]*dx[1]*dx[2]);

                for (int l = 0; l < AMREX_SPACEDIM; l++) {
                    for (int m = 0; m < AMREX_SPACEDIM; m++) {
                        amrex::Real tmp = 0.0;
                        for (int n=0; n < AMREX_SPACEDIM; n++)
                            tmp -= cs * cs * ds * ds * 0.25
                                * ( c1 * (gv[l*AMREX_SPACEDIM+n]+gv[n*AMREX_SPACEDIM+l])*(gv[n*AMREX_SPACEDIM+m] + gv[m*AMREX_SPACEDIM+n])
                                    + c2 * (gv[l*AMREX_SPACEDIM+n]+gv[n*AMREX_SPACEDIM+l])*(gv[n*AMREX_SPACEDIM+m] - gv[m*AMREX_SPACEDIM+n])
                                    - c2 * (gv[l*AMREX_SPACEDIM+n]-gv[n*AMREX_SPACEDIM+l])*(gv[n*AMREX_SPACEDIM+m] + gv[m*AMREX_SPACEDIM+n]) ) ;
                        mij_nonlin_arr(i, j, k, l*AMREX_SPACEDIM+m) = tmp;
                    }
                    mij_nonlin_arr(i, j, k, l*AMREX_SPACEDIM+l) += cs * cs * ds * ds * sijsij/3.0;
                }
                
            });
    }

    FTypeOut& m_mij_nonlin;
    const Field& m_vel;
    const amrex::Real m_cs;
    const amrex::Real m_c1;
    const amrex::Real m_c2;
};

template<typename FTypeOut>
inline void kc2000_nonlin_stress(FTypeOut& mij_nonlin,
                                 const Field& vel,
                                 amrex::Real cs,
                                 amrex::Real c1,
                                 amrex::Real c2 )
{
    KC2000NonlinStress<FTypeOut> grad(mij_nonlin, vel, cs, c1, c2);
    amr_wind::fvm::impl::apply(grad, vel);
}

inline std::unique_ptr<ScratchField> kc2000_nonlin_stress(const Field& vel,
                                                          amrex::Real cs,
                                                          amrex::Real c1,
                                                          amrex::Real c2 )
{
    const std::string gname = "mij_nonlin";
    auto mij_nonlin = vel.repo().create_scratch_field(
        gname, AMREX_SPACEDIM * AMREX_SPACEDIM);
    kc2000_nonlin_stress(*mij_nonlin, vel, cs, c1, c2);
    return mij_nonlin;
}

} // namespace fvm
} // namespace amr_wind

#endif /* KSGS_KC2000_UTILS_H */
