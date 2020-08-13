#include "amr-wind/turbulence/LES/OneEqKsgs.H"
#include "amr-wind/turbulence/LES/OneEqKsgsI.H"
#include "amr-wind/equation_systems/PDEBase.H"
#include "amr-wind/turbulence/TurbModelDefs.H"
#include "amr-wind/derive/derive_K.H"
#include "amr-wind/turbulence/turb_utils.H"
#include "amr-wind/equation_systems/tke/TKE.H"
#include "amr-wind/turbulence/LES/ksgs_kc2000_utils.H"
#include "amr-wind/fvm/divergence.H"

#include "AMReX_ParmParse.H"

namespace amr_wind {
namespace turbulence {

template <typename Transport>
KsgsKC2000<Transport>::KsgsKC2000(CFDSim& sim)
    : OneEqKsgs<Transport>(sim),
      m_temperature(sim.repo().get_field("temperature")),
      m_div_mij(sim.repo().declare_field("kc2000_div_nonlin_mij",3,1,1))
{

    auto& phy_mgr = this->m_sim.physics_manager();
    if (!phy_mgr.contains("ABL")) 
        amrex::Abort("KsgsKC2000 model only works with ABL physics");
        
    {
        const std::string coeffs_dict = this->model_name() + "_coeffs";
        amrex::ParmParse pp(coeffs_dict);
        pp.query("Ceps", this->m_Ceps);
        pp.query("Cb", this->m_Cb);
    }

    this->m_Cs = std::sqrt( 8.0 * (1.0 + this->m_Cb) / (27.0 * M_PI * M_PI));
    this->m_Ce = std::cbrt(8.0 * M_PI/27.0) * this->m_Cs * std::cbrt(this->m_Cs);
    this->m_C1 = std::sqrt(960.0 * this->m_Cb) / (7.0 * (1.0 + this->m_Cb) * 0.5);
    this->m_C2 = this->m_C1;
    
    {
        amrex::ParmParse pp("ABL");
        pp.get("reference_temperature", m_ref_theta);
    }

    {
        amrex::ParmParse pp("incflo");
        pp.queryarr("gravity", m_gravity);
    }
    
    // TKE source term to be added to PDE
    turb_utils::inject_turbulence_src_terms(pde::TKE::pde_name(), {"OneEqKsgsSrc"});
}

template <typename Transport>
KsgsKC2000<Transport>::~KsgsKC2000() = default;

template <typename Transport>
TurbulenceModel::CoeffsDictType KsgsKC2000<Transport>::model_coeffs() const
{
    return TurbulenceModel::CoeffsDictType{
        {"Ce", this->m_Ce},
        {"Ceps", this->m_Ceps},
        {"Cb", this->m_Cb},
        {"Cs", this->m_Cs},
        {"C1", this->m_C1},
        {"C2", this->m_C2}
    };
}

template <typename Transport>
void KsgsKC2000<Transport>::update_turbulent_viscosity(
    const FieldState fstate)
{
    BL_PROFILE("amr-wind::" + this->identifier() + "::update_turbulent_viscosity")

    auto gradT = (this->m_sim.repo()).create_scratch_field(1,0);
    compute_dir_gradient(*gradT, m_temperature, 2);

    auto& vel = this->m_vel.state(fstate);
    // Compute vertical gradient of velocity
    auto gradVelZ = (this->m_sim.repo()).create_scratch_field(3,0);
    compute_dir_gradient(*gradVelZ, vel, 2);
    // Compute strain rate into shear production term
    compute_strainrate(this->m_shear_prod, vel);
    
    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> gravity{
        m_gravity[0], m_gravity[1], m_gravity[2]};
    const amrex::Real beta = 1.0/m_ref_theta;
    const amrex::Real invPrandtl = 1.0/(this->m_transport.turbulent_prandtl());
    
    auto& mu_turb = this->mu_turb();
    const amrex::Real Ce = this->m_Ce;
    const amrex::Real Ceps = this->m_Ceps;
    const amrex::Real Cs = this->m_Cs;
    const amrex::Real C1 = this->m_C1;
    const amrex::Real C2 = this->m_C2;
    
    auto& den = this->m_rho.state(fstate);
    auto& repo = mu_turb.repo();
    auto& geom_vec = repo.mesh().Geom();

    const int nlevels = repo.num_active_levels();
    for (int lev=0; lev < nlevels; ++lev) {
        const auto& geom = geom_vec[lev];

        const amrex::Real dx = geom.CellSize()[0];
        const amrex::Real dy = geom.CellSize()[1];
        const amrex::Real dz = geom.CellSize()[2];
        const amrex::Real ds = std::cbrt(dx * dy * dz);

        for (amrex::MFIter mfi(mu_turb(lev)); mfi.isValid(); ++mfi) {
            const auto& bx = mfi.tilebox();
            const auto& mu_arr = mu_turb(lev).array(mfi);
            const auto& rho_arr = den(lev).const_array(mfi);
            const auto& gradT_arr = (*gradT)(lev).array(mfi);
            const auto& tke_arr = (*this->m_tke)(lev).array(mfi);
            const auto& buoy_prod_arr = (this->m_buoy_prod)(lev).array(mfi);
            const auto& shear_prod_arr = (this->m_shear_prod)(lev).array(mfi);
            const auto& sfs_ke_diss_arr = (this->m_sfs_ke_diss)(lev).array(mfi);
            const auto& gradVelZ_arr = (*gradVelZ)(lev).array(mfi);
            
            amrex::ParallelFor(
                bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {

                  mu_arr(i, j, k) =
                      rho_arr(i, j, k) * Ce
                      * ds * std::sqrt(tke_arr(i,j,k));

                  const amrex::Real strat
                      = gradT_arr(i,j,k) * gravity[2]* beta;
                  
                  buoy_prod_arr(i,j,k) =
                      mu_arr(i,j,k) * invPrandtl * strat;

                  const amrex::Real tmp1 =
                      ( (strat*strat/(0.76*0.76))
                        + (gradVelZ_arr(i,j,k,0)*gradVelZ_arr(i,j,k,0)
                           +gradVelZ_arr(i,j,k,1)*gradVelZ_arr(i,j,k,1))
                        /(2.76 * 2.76) )
                      / (tke_arr(i,j,k)+1e-15);
                  const amrex::Real invLe = std::sqrt(1.0/(ds*ds) + tmp1);

                  sfs_ke_diss_arr(i,j,k) = -Ceps * std::sqrt(tke_arr(i,j,k)) *
                      tke_arr(i,j,k) * invLe;

                  shear_prod_arr(i,j,k) *= shear_prod_arr(i,j,k) * mu_arr(i,j,k);
                  
            });
        }
    }

    auto mij = kc2000_nonlin_stress(vel, Cs, C1, C2);
    mij->fillpatch(this->m_sim.time().current_time());
    amr_wind::fvm::divergence(m_div_mij, *mij);
    
}


} // namespace turbulence

INSTANTIATE_TURBULENCE_MODEL(KsgsKC2000);

} // namespace amr_wind
