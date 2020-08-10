#include <cmath>

#include "amr-wind/turbulence/LES/AMD.H"
#include "amr-wind/turbulence/TurbModelDefs.H"
#include "amr-wind/derive/derive_K.H"

#include "AMReX_REAL.H"
#include "AMReX_MultiFab.H"
#include "AMReX_ParmParse.H"

namespace amr_wind {
namespace turbulence {

template<typename Transport>
AMD<Transport>::AMD(CFDSim& sim)
    : TurbModelBase<Transport>(sim)
    , m_vel(sim.repo().get_field("velocity"))
    , m_temperature(sim.repo().get_field("temperature"))
    , m_rho(sim.repo().get_field("density"))
{
    const std::string coeffs_dict = this->model_name() + "_coeffs";
    amrex::ParmParse pp(coeffs_dict);
    pp.query("C_poincare", m_C);

    auto& phy_mgr = this->m_sim.physics_manager();
    if (!phy_mgr.contains("ABL")) 
        amrex::Abort("OneEqKsgsM84 model only works with ABL physics");
    
    {
        amrex::ParmParse pp("ABL");
        pp.get("reference_temperature", m_ref_theta);
    }
    
}

template <typename Transport>
void AMD<Transport>::update_turbulent_viscosity(const FieldState fstate)
{
    BL_PROFILE("amr-wind::" + this->identifier() + "::update_turbulent_viscosity");

    auto& mu_turb = this->mu_turb();
    auto& repo = mu_turb.repo();
    auto& vel = m_vel.state(fstate);
    auto& temp = m_temperature.state(fstate);
    auto& den = m_rho.state(fstate);
    auto& geom_vec = repo.mesh().Geom();
    const amrex::Real beta = 1.0/m_ref_theta;
    const amrex::Real C_poincare = this->m_C;
        
    const int nlevels = repo.num_active_levels();
    for (int lev=0; lev < nlevels; ++lev) {
        const auto& geom = geom_vec[lev];
        const auto& domain = geom.Domain();

        const amrex::Real dx = geom.CellSize()[0];
        const amrex::Real dy = geom.CellSize()[1];
        const amrex::Real dz = geom.CellSize()[2];

        for (amrex::MFIter mfi(mu_turb(lev)); mfi.isValid(); ++mfi) {
            const auto& bx = mfi.tilebox();
            const auto& mu_arr = mu_turb(lev).array(mfi);
            const auto& rho_arr = den(lev).const_array(mfi);
            const auto& vel_arr = vel(lev).array(mfi);
            const auto& temp_arr = temp(lev).array(mfi);
                
            amrex::ParallelFor(
                bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                  const amrex::Real rho = rho_arr(i, j, k);
                  mu_arr(i, j, k) =
                      rho * amd_muvel<StencilInterior>
                      (i,j,k,dx,dy,dz,beta,C_poincare,vel_arr,temp_arr);
            });

            // TODO: Check if the following is correct for `foextrap` BC types
            const auto& bxi = mfi.tilebox();
            int idim = 0;
            if (!geom.isPeriodic(idim)) {
                if (bxi.smallEnd(idim) == domain.smallEnd(idim)) {
                    amrex::IntVect low(bxi.smallEnd());
                    amrex::IntVect hi(bxi.bigEnd());
                    int sm = low[idim];
                    low.setVal(idim, sm);
                    hi.setVal(idim, sm);
                    
                    auto bxlo = amrex::Box(low, hi);
                    
                    amrex::ParallelFor(
                        bxlo, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                                  const amrex::Real rho = rho_arr(i, j, k);
                                  mu_arr(i, j, k) =
                                      rho * amd_muvel<StencilILO>
                                      (i,j,k,dx,dy,dz,beta,C_poincare,vel_arr,temp_arr);
                              });
                }
                
                if (bxi.bigEnd(idim) == domain.bigEnd(idim)) {
                    amrex::IntVect low(bxi.smallEnd());
                    amrex::IntVect hi(bxi.bigEnd());
                    int sm = hi[idim];
                    low.setVal(idim, sm);
                    hi.setVal(idim, sm);
                    
                    auto bxhi = amrex::Box(low, hi);
                    
                    amrex::ParallelFor(
                        bxhi, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                                  const amrex::Real rho = rho_arr(i, j, k);
                                  mu_arr(i, j, k) =
                                      rho * amd_muvel<StencilIHI>
                                      (i,j,k,dx,dy,dz,beta,C_poincare,vel_arr,temp_arr);
                              });
                }
            } // if (!geom.isPeriodic)
            
            idim = 1;
            if (!geom.isPeriodic(idim)) {
                if (bxi.smallEnd(idim) == domain.smallEnd(idim)) {
                    amrex::IntVect low(bxi.smallEnd());
                    amrex::IntVect hi(bxi.bigEnd());
                    int sm = low[idim];
                    low.setVal(idim, sm);
                    hi.setVal(idim, sm);
                    
                    auto bxlo = amrex::Box(low, hi);
                    
                    amrex::ParallelFor(
                        bxlo, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                                  const amrex::Real rho = rho_arr(i, j, k);
                                  mu_arr(i, j, k) =
                                      rho * amd_muvel<StencilJLO>
                                      (i,j,k,dx,dy,dz,beta,C_poincare,vel_arr,temp_arr);
                              });
                }
                
                if (bxi.bigEnd(idim) == domain.bigEnd(idim)) {
                    amrex::IntVect low(bxi.smallEnd());
                    amrex::IntVect hi(bxi.bigEnd());
                    int sm = hi[idim];
                    low.setVal(idim, sm);
                    hi.setVal(idim, sm);
                    
                    auto bxhi = amrex::Box(low, hi);
                    
                    amrex::ParallelFor(
                        bxhi, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                                  const amrex::Real rho = rho_arr(i, j, k);
                                  mu_arr(i, j, k) =
                                      rho * amd_muvel<StencilJHI>
                                      (i,j,k,dx,dy,dz,beta,C_poincare,vel_arr,temp_arr);
                              });
                }
            } // if (!geom.isPeriodic)
            
            idim = 2;
            if (!geom.isPeriodic(idim)) {
                if (bxi.smallEnd(idim) == domain.smallEnd(idim)) {
                    amrex::IntVect low(bxi.smallEnd());
                    amrex::IntVect hi(bxi.bigEnd());
                    int sm = low[idim];
                    low.setVal(idim, sm);
                    hi.setVal(idim, sm);
                    
                    auto bxlo = amrex::Box(low, hi);
                    
                    amrex::ParallelFor(
                        bxlo, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                                  const amrex::Real rho = rho_arr(i, j, k);
                                  mu_arr(i, j, k) =
                                      rho * amd_muvel<StencilKLO>
                                      (i,j,k,dx,dy,dz,beta,C_poincare,vel_arr,temp_arr);
                              });
                }
                
                if (bxi.bigEnd(idim) == domain.bigEnd(idim)) {
                    amrex::IntVect low(bxi.smallEnd());
                    amrex::IntVect hi(bxi.bigEnd());
                    int sm = hi[idim];
                    low.setVal(idim, sm);
                    hi.setVal(idim, sm);
                    
                    auto bxhi = amrex::Box(low, hi);
                    
                    amrex::ParallelFor(
                        bxhi, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                                  const amrex::Real rho = rho_arr(i, j, k);
                                  mu_arr(i, j, k) =
                                      rho * amd_muvel<StencilKHI>
                                      (i,j,k,dx,dy,dz,beta,C_poincare,vel_arr,temp_arr);
                              });
                }
            } // if (!geom.isPeriodic)
        }
    }

    mu_turb.fillpatch(this->m_sim.time().current_time());
}

//! Update the effective thermal diffusivity field
template<typename Transport>
void AMD<Transport>::update_alphaeff(Field& alphaeff) {

    BL_PROFILE("amr-wind::" + this->identifier() + "::update_turbulent_viscosity");

    auto& repo = alphaeff.repo();
    auto& geom_vec = repo.mesh().Geom();
    const amrex::Real C_poincare = this->m_C;
        
    const int nlevels = repo.num_active_levels();
    for (int lev=0; lev < nlevels; ++lev) {
        const auto& geom = geom_vec[lev];
        const auto& domain = geom.Domain();

        const amrex::Real dx = geom.CellSize()[0];
        const amrex::Real dy = geom.CellSize()[1];
        const amrex::Real dz = geom.CellSize()[2];

        for (amrex::MFIter mfi(alphaeff(lev)); mfi.isValid(); ++mfi) {
            const auto& bx = mfi.tilebox();
            const auto& alpha_arr = alphaeff(lev).array(mfi);
            const auto& rho_arr = m_rho(lev).const_array(mfi);
            const auto& vel_arr = m_vel(lev).array(mfi);
            const auto& temp_arr = m_temperature(lev).array(mfi);
                
            amrex::ParallelFor(
                bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                  const amrex::Real rho = rho_arr(i, j, k);
                  alpha_arr(i, j, k) =
                      rho * amd_thermal_diff<StencilInterior>
                      (i,j,k,dx,dy,dz,C_poincare,vel_arr,temp_arr);
            });


            // TODO: Check if the following is correct for `foextrap` BC types
            const auto& bxi = mfi.tilebox();
            int idim = 0;
            if (!geom.isPeriodic(idim)) {
                if (bxi.smallEnd(idim) == domain.smallEnd(idim)) {
                    amrex::IntVect low(bxi.smallEnd());
                    amrex::IntVect hi(bxi.bigEnd());
                    int sm = low[idim];
                    low.setVal(idim, sm);
                    hi.setVal(idim, sm);
                    
                    auto bxlo = amrex::Box(low, hi);
                    
                    amrex::ParallelFor(
                        bxlo, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                                  const amrex::Real rho = rho_arr(i, j, k);
                                  alpha_arr(i, j, k) =
                                      rho * amd_thermal_diff<StencilILO>
                                      (i,j,k,dx,dy,dz,C_poincare,vel_arr,temp_arr);
                              });
                }
                
                if (bxi.bigEnd(idim) == domain.bigEnd(idim)) {
                    amrex::IntVect low(bxi.smallEnd());
                    amrex::IntVect hi(bxi.bigEnd());
                    int sm = hi[idim];
                    low.setVal(idim, sm);
                    hi.setVal(idim, sm);
                    
                    auto bxhi = amrex::Box(low, hi);
                    
                    amrex::ParallelFor(
                        bxhi, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                                  const amrex::Real rho = rho_arr(i, j, k);
                                  alpha_arr(i, j, k) =
                                      rho * amd_thermal_diff<StencilIHI>
                                      (i,j,k,dx,dy,dz,C_poincare,vel_arr,temp_arr);
                              });
                }
            } // if (!geom.isPeriodic)
            
            idim = 1;
            if (!geom.isPeriodic(idim)) {
                if (bxi.smallEnd(idim) == domain.smallEnd(idim)) {
                    amrex::IntVect low(bxi.smallEnd());
                    amrex::IntVect hi(bxi.bigEnd());
                    int sm = low[idim];
                    low.setVal(idim, sm);
                    hi.setVal(idim, sm);
                    
                    auto bxlo = amrex::Box(low, hi);
                    
                    amrex::ParallelFor(
                        bxlo, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                                  const amrex::Real rho = rho_arr(i, j, k);
                                  alpha_arr(i, j, k) =
                                      rho * amd_thermal_diff<StencilJLO>
                                      (i,j,k,dx,dy,dz,C_poincare,vel_arr,temp_arr);
                              });
                }
                
                if (bxi.bigEnd(idim) == domain.bigEnd(idim)) {
                    amrex::IntVect low(bxi.smallEnd());
                    amrex::IntVect hi(bxi.bigEnd());
                    int sm = hi[idim];
                    low.setVal(idim, sm);
                    hi.setVal(idim, sm);
                    
                    auto bxhi = amrex::Box(low, hi);
                    
                    amrex::ParallelFor(
                        bxhi, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                                  const amrex::Real rho = rho_arr(i, j, k);
                                  alpha_arr(i, j, k) =
                                      rho * amd_thermal_diff<StencilJHI>
                                      (i,j,k,dx,dy,dz,C_poincare,vel_arr,temp_arr);
                              });
                }
            } // if (!geom.isPeriodic)
            
            idim = 2;
            if (!geom.isPeriodic(idim)) {
                if (bxi.smallEnd(idim) == domain.smallEnd(idim)) {
                    amrex::IntVect low(bxi.smallEnd());
                    amrex::IntVect hi(bxi.bigEnd());
                    int sm = low[idim];
                    low.setVal(idim, sm);
                    hi.setVal(idim, sm);
                    
                    auto bxlo = amrex::Box(low, hi);
                    
                    amrex::ParallelFor(
                        bxlo, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                                  const amrex::Real rho = rho_arr(i, j, k);
                                  alpha_arr(i, j, k) =
                                      rho * amd_thermal_diff<StencilKLO>
                                      (i,j,k,dx,dy,dz,C_poincare,vel_arr,temp_arr);
                              });
                }
                
                if (bxi.bigEnd(idim) == domain.bigEnd(idim)) {
                    amrex::IntVect low(bxi.smallEnd());
                    amrex::IntVect hi(bxi.bigEnd());
                    int sm = hi[idim];
                    low.setVal(idim, sm);
                    hi.setVal(idim, sm);
                    
                    auto bxhi = amrex::Box(low, hi);
                    
                    amrex::ParallelFor(
                        bxhi, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                                  const amrex::Real rho = rho_arr(i, j, k);
                                  alpha_arr(i, j, k) =
                                      rho * amd_thermal_diff<StencilKHI>
                                      (i,j,k,dx,dy,dz,C_poincare,vel_arr,temp_arr);
                              });
                }
            } // if (!geom.isPeriodic)
        }
    }
    
}



template<typename Transport>
TurbulenceModel::CoeffsDictType AMD<Transport>::model_coeffs() const
{
    return TurbulenceModel::CoeffsDictType{
        {"C_poincare", this->m_C}
    };
}

} // namespace turbulence

INSTANTIATE_TURBULENCE_MODEL(AMD);

} // namespace amr_wind
