#include "OneEqKsgs.H"
#include "PDEBase.H"
#include "TurbModelDefs.H"
#include "derive_K.H"
#include "turb_utils.H"
#include "tke/TKE.H"
#include "icns/icns.H"
#include "FieldPlaneAveraging.H"
#include "SecondMomentAveraging.H"
#include "ThirdMomentAveraging.H"
#include "ABL.H"
#include "AMReX_ParmParse.H"
#include "DirectionSelector.H"

namespace amr_wind {
namespace turbulence {

template <typename Transport>
OneEqKsgs<Transport>::OneEqKsgs(CFDSim& sim)
    : TurbModelBase<Transport>(sim), m_vel(sim.repo().get_field("velocity")),
      m_turb_lscale(sim.repo().declare_field("turb_lscale",1, 1, 1)),
      m_shear_prod(sim.repo().declare_field("shear_prod",1, 1, 1)),
      m_buoy_prod(sim.repo().declare_field("buoy_prod",1, 1, 1)),
      m_rho(sim.repo().get_field("density"))
{
    auto& tke_eqn = sim.pde_manager().register_transport_pde(pde::TKE::pde_name());
    m_tke = &(tke_eqn.fields().field);

    //Turbulent length scale field
    this->m_sim.io_manager().register_io_var("turb_lscale");
}

template <typename Transport>
OneEqKsgs<Transport>::~OneEqKsgs() = default;

template <typename Transport>
OneEqKsgsM84<Transport>::OneEqKsgsM84(CFDSim& sim)
    : OneEqKsgs<Transport>(sim)
    , m_temperature(sim.repo().get_field("temperature"))
{

    auto& phy_mgr = this->m_sim.physics_manager();
    if (!phy_mgr.contains("ABL")) 
        amrex::Abort("OneEqKsgsM84 model only works with ABL physics");
        
    {
        const std::string coeffs_dict = this->model_name() + "_coeffs";
        amrex::ParmParse pp(coeffs_dict);
        pp.query("Ceps", this->m_Ceps);
        pp.query("Ce", this->m_Ce);
    }

    {
        amrex::ParmParse pp("ABL");
        pp.get("reference_temperature", m_ref_theta);
    }

    {
        amrex::ParmParse pp("incflo");
        pp.queryarr("gravity", m_gravity);
    }
    
    // TKE source term to be added to PDE
    turb_utils::inject_turbulence_src_terms(pde::TKE::pde_name(), {"KsgsM84Src"});
}

template <typename Transport>
OneEqKsgsM84<Transport>::~OneEqKsgsM84() = default;

template <typename Transport>
TurbulenceModel::CoeffsDictType OneEqKsgsM84<Transport>::model_coeffs() const
{
    return TurbulenceModel::CoeffsDictType{{"Ce", this->m_Ce}, {"Ceps", this->m_Ceps}};
}

template <typename Transport>
void OneEqKsgsM84<Transport>::update_turbulent_viscosity(
    const FieldState fstate)
{
    BL_PROFILE("amr-wind::" + this->identifier() + "::update_turbulent_viscosity")

    auto gradT = (this->m_sim.repo()).create_scratch_field(3,m_temperature.num_grow()[0]);
    compute_gradient(*gradT, m_temperature);

    auto& vel = this->m_vel.state(fstate);
    // Compute strain rate into shear production term
    compute_strainrate(this->m_shear_prod, vel);
    
    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> gravity{
        m_gravity[0], m_gravity[1], m_gravity[2]};
    const amrex::Real beta = 1.0/m_ref_theta;
    
    auto& mu_turb = this->mu_turb();
    const amrex::Real Ce = this->m_Ce;
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
            const auto& bx = mfi.growntilebox(mu_turb.num_grow());
            const auto& mu_arr = mu_turb(lev).array(mfi);
            const auto& rho_arr = den(lev).const_array(mfi);
            const auto& gradT_arr = (*gradT)(lev).array(mfi);
            const auto& tlscale_arr = (this->m_turb_lscale)(lev).array(mfi);
            const auto& tke_arr = (*this->m_tke)(lev).array(mfi);
            const auto& buoy_prod_arr = (this->m_buoy_prod)(lev).array(mfi);
            const auto& shear_prod_arr = (this->m_shear_prod)(lev).array(mfi);
            
            amrex::ParallelFor(
                bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {

                  amrex::Real stratification =
                      (gradT_arr(i,j,k,0) * gravity[0]
                       + gradT_arr(i,j,k,1) * gravity[1]
                       + gradT_arr(i,j,k,2) * gravity[2])*beta;
                  if(stratification > 1e-10)
                      tlscale_arr(i,j,k) =
                          0.76 * stratification * std::sqrt(tke_arr(i,j,k));
                  else
                      tlscale_arr(i,j,k) = ds;
                        
                  mu_arr(i, j, k) =
                      rho_arr(i, j, k) * Ce
                      * tlscale_arr(i,j,k) * std::sqrt(tke_arr(i,j,k));
                  
                  buoy_prod_arr(i,j,k) =
                      mu_arr(i,j,k) * (1.0 + 2.0 * tlscale_arr(i,j,k)/ds)
                      * stratification;

                  shear_prod_arr(i,j,k) *= mu_arr(i,j,k);
                  
                    });
        }
    }
    
}

template <typename Transport>
void OneEqKsgsM84<Transport>::update_alphaeff(Field& alphaeff)
{

    BL_PROFILE("amr-wind::" + this->identifier() + "::update_alphaeff")
    
    amrex::Real lam_diff = (this->m_transport).thermal_diffusivity();
    auto& mu_turb = this->m_mu_turb;
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
            const auto& bx = mfi.growntilebox(mu_turb.num_grow());
            const auto& muturb_arr = mu_turb(lev).array(mfi);
            const auto& alphaeff_arr = alphaeff(lev).array(mfi);
            const auto& tlscale_arr = this->m_turb_lscale(lev).array(mfi);
            
            amrex::ParallelFor(
                bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                        alphaeff_arr(i, j, k) = lam_diff + 
                            muturb_arr(i,j,k)
                            * (1.0 + 2.0 * tlscale_arr(i,j,k)/ds);
                    });
        }
    }
    
}

template <typename Transport>
OneEqKsgsS94<Transport>::OneEqKsgsS94(CFDSim& sim)
    : OneEqKsgs<Transport>(sim)
    , m_vel(sim.repo().get_field("velocity"))
    , m_temperature(sim.repo().get_field("temperature"))
    , m_turb_lscale(sim.repo().declare_field("turb_lscale",1, 1, 1))
    , m_shear_prod(sim.repo().declare_field("shear_prod",1, 1, 1))
    , m_buoy_prod(sim.repo().declare_field("buoy_prod",1, 1, 1))
    , m_nut(sim.repo().declare_field("nut",1, 1, 1))
    , m_nuT(sim.repo().declare_field("nuT",1, 1, 1))
    , m_wind_speed(sim.repo().declare_field("wind_speed",1, 1, 1))
    , m_abl(sim.physics_manager().get<amr_wind::ABL>())
    , m_ref_height(m_abl.log_law_height())
{
    
    auto& phy_mgr = this->m_sim.physics_manager();
    if (!phy_mgr.contains("ABL")) 
        amrex::Abort("OneEqKsgsS94 model only works with ABL physics");

    m_pa_vel = &(m_abl.abl_field_plane_averaging());
    
    {
        const std::string coeffs_dict = this->model_name() + "_coeffs";
        amrex::ParmParse pp(coeffs_dict);
        pp.query("Ceps", this->m_Ceps);
        pp.query("Ce", this->m_Ce);
    }

    {
        amrex::ParmParse pp("ABL");
        pp.get("reference_temperature", m_ref_theta);
    }

    {
        amrex::ParmParse pp("incflo");
        pp.queryarr("gravity", m_gravity);
    }

    // TKE source term to be added to PDE
    turb_utils::inject_turbulence_src_terms(pde::ICNS::pde_name(), {"MeanTurbDiffS94Src"});
    turb_utils::inject_turbulence_src_terms(pde::TKE::pde_name(), {"KsgsS94Src"});
    
}

template <typename Transport>
OneEqKsgsS94<Transport>::~OneEqKsgsS94() = default;

template <typename Transport>
TurbulenceModel::CoeffsDictType OneEqKsgsS94<Transport>::model_coeffs() const
{
    return TurbulenceModel::CoeffsDictType{{"Ce", this->m_Ce}, {"Ceps", this->m_Ceps}};
}

template <typename Transport>
void OneEqKsgsS94<Transport>::update_turbulent_viscosity(
    const FieldState fstate)
{

    BL_PROFILE("amr-wind::" + this->identifier() + "::update_turbulent_viscosity")

    /* Steps
       1. Compute strain rate
       2. Compute average strain rate
       3. Get average velocity and its gradient
       4. Compute S'
       5. Compute gamma, nut, shear production, buoyancy production
       term. Store nut * gamma in mu_turb and just nut in nut for
       later use
       7. Compute average of <nut gamma>
       8. Compute nuTstar
       9. Compute nuT
    */
        
    auto gradT = (this->m_sim.repo()).create_scratch_field(3,m_temperature.num_grow()[0]);
    compute_gradient(*gradT, m_temperature);

    auto& vel = this->m_vel.state(fstate);
    // Compute strain rate into shear production term
    compute_strainrate(this->m_shear_prod, vel);
    FieldPlaneAveraging avg_str_rate(this->m_shear_prod, this->m_sim.time(), 2);
    avg_str_rate();
    //Now compute fluctuating strain rate
    const auto& c_pa_vel = *m_pa_vel;
    auto& pa_vel = *m_pa_vel;
    compute_fluct_strainrate(this->m_shear_prod, vel, c_pa_vel, ZDir());
    SecondMomentAveraging uu(pa_vel, pa_vel);
    

    //TODO: Move this to ABL
        
    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> gravity{
        m_gravity[0], m_gravity[1], m_gravity[2]};
    const amrex::Real beta = 1.0/m_ref_theta;
    
    auto& mu_turb = this->mu_turb();
    const amrex::Real Ce = this->m_Ce;
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
            const auto& bx = mfi.growntilebox(mu_turb.num_grow());
            const auto& mu_arr = mu_turb(lev).array(mfi);
            const auto& nut_arr = (this->m_nut)(lev).array(mfi);
            const auto& rho_arr = den(lev).const_array(mfi);
            const auto& gradT_arr = (*gradT)(lev).array(mfi);
            const auto& tlscale_arr = (this->m_turb_lscale)(lev).array(mfi);
            const auto& tke_arr = (*this->m_tke)(lev).array(mfi);
            const auto& buoy_prod_arr = (this->m_buoy_prod)(lev).array(mfi);
            const auto& shear_prod_arr = (this->m_shear_prod)(lev).array(mfi);
            const auto& vel_arr = vel(lev).const_array(mfi);
            const auto& wind_speed_arr = m_wind_speed(lev).array(mfi);
            
            amrex::ParallelFor(
                bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {

              amrex::Real stratification =
                  (gradT_arr(i,j,k,0) * gravity[0]
                   + gradT_arr(i,j,k,1) * gravity[1]
                   + gradT_arr(i,j,k,2) * gravity[2])*beta;
              if(stratification > 1e-10)
                  tlscale_arr(i,j,k) =
                      0.76 * stratification * std::sqrt(tke_arr(i,j,k));
              else
                  tlscale_arr(i,j,k) = ds;
                        
              nut_arr(i, j, k) =
                  rho_arr(i, j, k) * Ce
                  * tlscale_arr(i,j,k) * std::sqrt(tke_arr(i,j,k));
              
              buoy_prod_arr(i,j,k) =
                  mu_arr(i,j,k) * (1.0 + 2.0 * tlscale_arr(i,j,k)/ds)
                  * stratification;

              amrex::Real gamma = shear_prod_arr(i,j,k)
                  / (shear_prod_arr(i,j,k)
                     + m_pa_vel->line_average_cell(k,0));

              mu_arr(i,j,k) = nut_arr(i,j,k) * gamma;

              shear_prod_arr(i,j,k) *=
                  shear_prod_arr(i,j,k) * nut_arr(i,j,k);

              wind_speed_arr(i,j,k) =
                  std::sqrt(
                      vel_arr(i,j,k,0)*vel_arr(i,j,k,0)
                      + vel_arr(i,j,k,1)*vel_arr(i,j,k,1) );
              
            });
        }
    }

    FieldPlaneAveraging nut_gamma(mu_turb, this->m_sim.time(), 2);
    nut_gamma();
    
    FieldPlaneAveraging wind_speed_avg(m_wind_speed, this->m_sim.time(), 2);
    wind_speed_avg();
    
    amrex::Real utau = m_abl.utau();
    const auto& dx = geom_vec[0].CellSizeArray();
    const auto& problo = geom_vec[0].ProbLoArray();
    int ref_height_index = std::nearbyint(m_ref_height/dx[2] - 0.5);
    //TODO: Fix this to work with grid refinement
    amrex::Real dusdz_wf =
        wind_speed_avg.line_derivative_of_average_cell(ref_height_index, 0);
    amrex::Real upwp = uu.line_average_interpolated(m_ref_height, 3);
    amrex::Real vpwp = uu.line_average_interpolated(m_ref_height, 6);
    amrex::Real rho = 1.0; //TODO: Fix me - May be this should be an average density at m_ref_height
    m_nuT_star = rho * (-1.0/dusdz_wf * std::sqrt(upwp + vpwp) + utau * utau/dusdz_wf )
        - nut_gamma.line_average_interpolated(m_ref_height, 0);
    
    for (int lev=0; lev < nlevels; ++lev) {
        const auto& geom = geom_vec[lev];

        const amrex::Real dx = geom.CellSize()[0];
        const amrex::Real dy = geom.CellSize()[1];
        const amrex::Real dz = geom.CellSize()[2];
        const amrex::Real ds = std::cbrt(dx * dy * dz);

        for (amrex::MFIter mfi(mu_turb(lev)); mfi.isValid(); ++mfi) {
            const auto& bx = mfi.growntilebox(mu_turb.num_grow());
            const auto& nut_arr = mu_turb(lev).array(mfi);
            const auto& nuT_arr = m_nuT(lev).array(mfi);
            
            amrex::ParallelFor(
                bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {

              //TODO: Fix this to work with grid refinement
              amrex::Real umeanz = m_pa_vel->line_derivative_of_average_cell(k, 0);
              amrex::Real vmeanz = m_pa_vel->line_derivative_of_average_cell(k, 1);
              nuT_arr(i,j,k) = m_nuT_star / dusdz_wf 
                  * std::sqrt(umeanz*umeanz + vmeanz*vmeanz);
            });
        }
    }    
}

template <typename Transport>
void OneEqKsgsS94<Transport>::update_alphaeff(Field& alphaeff)
{

    BL_PROFILE("amr-wind::" + this->identifier() + "::update_alphaeff")
    
    amrex::Real lam_diff = (this->m_transport).thermal_diffusivity();
    auto& nut = this->m_nut;
    auto& repo = nut.repo();
    auto& geom_vec = repo.mesh().Geom();

    const int nlevels = repo.num_active_levels();
    for (int lev=0; lev < nlevels; ++lev) {
        const auto& geom = geom_vec[lev];

        const amrex::Real dx = geom.CellSize()[0];
        const amrex::Real dy = geom.CellSize()[1];
        const amrex::Real dz = geom.CellSize()[2];
        const amrex::Real ds = std::cbrt(dx * dy * dz);

        for (amrex::MFIter mfi(nut(lev)); mfi.isValid(); ++mfi) {
            const auto& bx = mfi.growntilebox(nut.num_grow());
            const auto& nut_arr = nut(lev).array(mfi);
            const auto& alphaeff_arr = alphaeff(lev).array(mfi);
            const auto& tlscale_arr = this->m_turb_lscale(lev).array(mfi);
            
            amrex::ParallelFor(
                bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                        alphaeff_arr(i, j, k) = lam_diff + 
                            nut_arr(i,j,k)
                            * (1.0 + 2.0 * tlscale_arr(i,j,k)/ds);
                    });
        }
    }
    
}


} // namespace turbulence

INSTANTIATE_TURBULENCE_MODEL(OneEqKsgsM84);
INSTANTIATE_TURBULENCE_MODEL(OneEqKsgsS94);

} // namespace amr_wind
