#include "amr-wind/turbulence/LES/OneEqKsgs.H"
#include "amr-wind/turbulence/LES/OneEqKsgsI.H"
#include "amr-wind/equation_systems/PDEBase.H"
#include "amr-wind/turbulence/TurbModelDefs.H"
#include "amr-wind/derive/derive_K.H"
#include "amr-wind/turbulence/turb_utils.H"
#include "amr-wind/equation_systems/tke/TKE.H"

#include "amr-wind/equation_systems/icns/icns.H"
#include "amr-wind/utilities/SecondMomentAveraging.H"
#include "amr-wind/utilities/DirectionSelector.H"

#include "AMReX_ParmParse.H"

namespace amr_wind {
namespace turbulence {

template <typename Transport>
OneEqKsgsM84<Transport>::OneEqKsgsM84(CFDSim& sim)
    : OneEqKsgs<Transport>(sim),
      m_temperature(sim.repo().get_field("temperature"))
{

    auto& phy_mgr = this->m_sim.physics_manager();
    if (!phy_mgr.contains("ABL")) 
        amrex::Abort("OneEqKsgsM84 model only works with ABL physics");
        
    {
        const std::string coeffs_dict = this->model_name() + "_coeffs";
        amrex::ParmParse pp(coeffs_dict);
        pp.query("Ceps", this->m_Ceps);
        pp.query("Ce", this->m_Ce);
        this->m_Ceps_ground =  (3.9/0.93) * (this->m_Ceps);
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

    auto gradT = (this->m_sim.repo()).create_scratch_field(3,0);
    compute_gradient(*gradT, m_temperature.state(fstate) );

    auto& vel = this->m_vel.state(fstate);
    // Compute strain rate into shear production term
    compute_strainrate(this->m_shear_prod, vel);
    
    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> gravity{
        m_gravity[0], m_gravity[1], m_gravity[2]};
    const amrex::Real beta = 1.0/m_ref_theta;
    
    auto& mu_turb = this->mu_turb();
    const amrex::Real Ce = this->m_Ce;
    const amrex::Real Ceps = this->m_Ceps;
    const amrex::Real CepsGround = this->m_Ceps_ground;
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
            const auto& tlscale_arr = (this->m_turb_lscale)(lev).array(mfi);
            const auto& tke_arr = (*this->m_tke)(lev).array(mfi);
            const auto& buoy_prod_arr = (this->m_buoy_prod)(lev).array(mfi);
            const auto& shear_prod_arr = (this->m_shear_prod)(lev).array(mfi);
            const auto& sfs_ke_diss_arr = (this->m_sfs_ke_diss)(lev).array(mfi);
            
            amrex::ParallelFor(
                bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {

                  amrex::Real stratification =
                      -(gradT_arr(i,j,k,0) * gravity[0]
                       + gradT_arr(i,j,k,1) * gravity[1]
                       + gradT_arr(i,j,k,2) * gravity[2])*beta;
                  if(stratification > 1e-10)
                      tlscale_arr(i,j,k) =
                          amrex::min(ds,
                                     0.76 * std::sqrt(tke_arr(i,j,k) / stratification) );
                  else
                      tlscale_arr(i,j,k) = ds;

                  mu_arr(i, j, k) =
                      rho_arr(i, j, k) * Ce
                      * tlscale_arr(i,j,k) * std::sqrt(tke_arr(i,j,k));
                  
                  buoy_prod_arr(i,j,k) =
                      -mu_arr(i,j,k) * (1.0 + 2.0 * tlscale_arr(i,j,k)/ds)
                       * stratification;

                  shear_prod_arr(i,j,k) *= shear_prod_arr(i,j,k) * mu_arr(i,j,k);

                  sfs_ke_diss_arr(i,j,k) = -Ceps * std::sqrt(tke_arr(i,j,k)) *
                      tke_arr(i,j,k) / tlscale_arr(i,j,k);
                  
            });

            auto& bctype = (*this->m_tke).bc_type();
            for (int dir=0; dir < AMREX_SPACEDIM; ++dir) {
                amrex::Orientation olo(dir, amrex::Orientation::low);
                if (bctype[olo] == BC::wall_model) {
                    amrex::Box blo = amrex::adjCellLo(bx,dir,0) & bx;
                    if (blo.ok()) {
                        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                          sfs_ke_diss_arr(i, j, k) = -CepsGround * std::sqrt(tke_arr(i,j,k)) *
                              tke_arr(i,j,k) / tlscale_arr(i,j,k);
                        });
                    }
                }
            }
            
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
            const auto& bx = mfi.tilebox();
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
void OneEqKsgsM84<Transport>::update_scalar_diff(Field& scalar_diff, const std::string& name )
{
    BL_PROFILE("amr-wind::" + this->identifier() + "::update_scalar_diff")

    if (name == "tke") {
        field_ops::saxpy(
            scalar_diff, 2.0, this->mu_turb(), 0, 0, scalar_diff.num_comp(), scalar_diff.num_grow());    
    } else {
        std::cerr << "Unknown scalar to compute diffusivity" << std::endl;
    }
}


template <typename Transport>
OneEqKsgsS94<Transport>::OneEqKsgsS94(CFDSim& sim)
    : OneEqKsgs<Transport>(sim)
    , m_vel(sim.repo().get_field("velocity"))
    , m_temperature(sim.repo().get_field("temperature"))
    , m_turb_lscale(sim.repo().declare_field("turb_lscale",1, 0, 1))
    , m_shear_prod(sim.repo().declare_field("shear_prod",1, 0, 1))
    , m_buoy_prod(sim.repo().declare_field("buoy_prod",1, 0, 1))
    , m_gamma(sim.repo().declare_field("gamma",1,1,1))
    , m_nuT(sim.repo().declare_field("nuT",1, 1, 1))
    , m_gradT(sim.repo().declare_field("gradT",3,1,1))
    , m_mean_stress_div(sim.repo().declare_field("s94_mean_stress_div",3,0,1))
    , m_abl(sim.physics_manager().get<amr_wind::ABL>())
    , m_ref_height(m_abl.abl_wall_function().log_law_height())
    , m_pa_muturb(this->mu_turb(), sim.time(), 2)
{
    
    auto& phy_mgr = this->m_sim.physics_manager();
    if (!phy_mgr.contains("ABL")) 
        amrex::Abort("OneEqKsgsS94 model only works with ABL physics");
    
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

    m_buoy_prod.set_default_fillpatch_bc(this->m_sim.time());
    m_shear_prod.set_default_fillpatch_bc(this->m_sim.time());
    m_gamma.set_default_fillpatch_bc(this->m_sim.time());
    m_gradT.set_default_fillpatch_bc(this->m_sim.time());
    
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
       1. Compute fluctuating strain rate
       2. Compute gamma, nut, shear production, buoyancy production
       term. Store nut * gamma in mu_turb 
       3. Compute average of <nut gamma>
       4. Compute nuTstar
       5. Compute nuT and mean stress divergence
    */

    const auto& abl_stats = m_abl.abl_stats(); //Get ABLStats object

    (this->m_temperature).fillpatch(this->m_sim.time().current_time());    
    //auto gradT = (this->m_sim.repo()).create_scratch_field(3,1)  ;
    compute_gradient(m_gradT, m_temperature);

    auto& vel = this->m_vel.state(fstate);
    //Compute fluctuating strain rate
    const auto& pa_vel = abl_stats.vel_plane_averaging();
    compute_fluct_strainrate(this->m_shear_prod, vel, pa_vel, ZDir::dir, ZDir());
    
    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> gravity{
        m_gravity[0], m_gravity[1], m_gravity[2]};
    const amrex::Real beta = 1.0/m_ref_theta;
    
    auto& mu_turb = this->mu_turb();
    const amrex::Real Ce = this->m_Ce;
    const amrex::Real Ceps = this->m_Ceps;
    const amrex::Real CepsGround = this->m_Ceps_ground;
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
            const auto& gamma_arr = (this->m_gamma)(lev).array(mfi);
            const auto& gradT_arr = (this->m_gradT)(lev).array(mfi);
            const auto& tlscale_arr = (this->m_turb_lscale)(lev).array(mfi);
            const auto& tke_arr = (*this->m_tke)(lev).array(mfi);
            const auto& buoy_prod_arr = (this->m_buoy_prod)(lev).array(mfi);
            const auto& shear_prod_arr = (this->m_shear_prod)(lev).array(mfi);
            const auto& sfs_ke_diss_arr = (this->m_sfs_ke_diss)(lev).array(mfi);

            amrex::ParallelFor(
                bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {

              amrex::Real stratification =
                  -(gradT_arr(i,j,k,0) * gravity[0]
                   + gradT_arr(i,j,k,1) * gravity[1]
                   + gradT_arr(i,j,k,2) * gravity[2])*beta;
              if(stratification > 1e-10)
                  tlscale_arr(i,j,k) =
                      amrex::min(ds, 0.76 * std::sqrt(tke_arr(i,j,k)/stratification));
              else
                  tlscale_arr(i,j,k) = ds;

              amrex::Real mut = rho_arr(i, j, k) * Ce
                  * tlscale_arr(i,j,k) * std::sqrt(tke_arr(i,j,k));

              buoy_prod_arr(i,j,k) =
                  -mut * (1.0 + 2.0 * tlscale_arr(i,j,k)/ds)
                  * stratification;
              
              sfs_ke_diss_arr(i,j,k) = -Ceps * std::sqrt(tke_arr(i,j,k)) *
                  tke_arr(i,j,k) / tlscale_arr(i,j,k);
              

              amrex::Real umeanz = pa_vel.line_derivative_of_average_cell(k, 0);
              amrex::Real vmeanz = pa_vel.line_derivative_of_average_cell(k, 1);
              gamma_arr(i,j,k) = amrex::max(1.0, shear_prod_arr(i,j,k)
                                          / (shear_prod_arr(i,j,k)
                                             + std::sqrt(umeanz*umeanz + vmeanz*vmeanz)
                                             + 1.0e-15) );

              mu_arr(i,j,k) = mut * gamma_arr(i,j,k);

              shear_prod_arr(i,j,k) *=
                  shear_prod_arr(i,j,k) * mu_arr(i,j,k);
              
            });

            auto& bctype = (*this->m_tke).bc_type();
            for (int dir=0; dir < AMREX_SPACEDIM; ++dir) {
                amrex::Orientation olo(dir, amrex::Orientation::low);
                if (bctype[olo] == BC::wall_model) {
                    amrex::Box blo = amrex::adjCellLo(bx,dir,0) & bx;
                    if (blo.ok()) {
                        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                          sfs_ke_diss_arr(i, j, k) = -CepsGround * std::sqrt(tke_arr(i,j,k)) *
                              tke_arr(i,j,k) / tlscale_arr(i,j,k);
                        });
                    }
                }
            }
            
        }
    }

    (this->mu_turb()).fillpatch(this->m_sim.time().current_time());
    (this->m_buoy_prod).fillpatch(this->m_sim.time().current_time());
    (this->m_gamma).fillpatch(this->m_sim.time().current_time());
    (this->m_shear_prod).fillpatch(this->m_sim.time().current_time());
    m_pa_muturb();
    
    amrex::Real utau = m_abl.abl_wall_function().utau();
    const auto& dx = geom_vec[0].CellSizeArray();
    int ref_height_index = std::nearbyint(m_ref_height/dx[2] - 0.5);
    //TODO: Fix this to work with grid refinement
    amrex::Real dusdz_wf =
        pa_vel.line_hvelmag_derivative_of_average_cell(ref_height_index);
    const auto& pa_uu = abl_stats.vel_uu_plane_averaging();
    amrex::Real upwp = pa_uu.line_average_interpolated(m_ref_height, 3);
    amrex::Real vpwp = pa_uu.line_average_interpolated(m_ref_height, 6);
    amrex::Real rho = 1.0; //TODO: Fix me - May be this should be an average density at m_ref_height
    m_nuT_star = rho * (utau*utau - std::sqrt(upwp*upwp + vpwp*vpwp) )/dusdz_wf
        - m_pa_muturb.line_average_interpolated(m_ref_height, 0);
    
    for (int lev=0; lev < nlevels; ++lev) {
        for (amrex::MFIter mfi(mu_turb(lev)); mfi.isValid(); ++mfi) {
            const auto& bx = mfi.tilebox();
            const auto& nuT_arr = m_nuT(lev).array(mfi);
            const auto& gradT_arr = (this->m_gradT)(lev).array(mfi); //Reuse
            
            amrex::ParallelFor(
                bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {

              //TODO: Fix this to work with grid refinement
              amrex::Real umeanz = pa_vel.line_derivative_of_average_cell(k, 0);
              amrex::Real vmeanz = pa_vel.line_derivative_of_average_cell(k, 1);
              nuT_arr(i,j,k) = m_nuT_star / dusdz_wf 
                  * std::sqrt(umeanz*umeanz + vmeanz*vmeanz);
              gradT_arr(i,j,k,0) = nuT_arr(i,j,k) * umeanz;
              gradT_arr(i,j,k,1) = nuT_arr(i,j,k) * vmeanz;
              gradT_arr(i,j,k,2) = 0.0;
              
            }); 
        }
    }

    (this->m_gradT).fillpatch(this->m_sim.time().current_time());
    (this->m_mean_stress_div).setVal(0.0);
    compute_dir_gradient((this->m_mean_stress_div), (this->m_gradT), 2);

    compute_gradient(m_gradT, m_temperature);
    
}

template <typename Transport>
void OneEqKsgsS94<Transport>::update_alphaeff(Field& alphaeff)
{

    BL_PROFILE("amr-wind::" + this->identifier() + "::update_alphaeff")
    
    amrex::Real lam_diff = (this->m_transport).thermal_diffusivity();
    const amrex::Real Ce = this->m_Ce;
    auto& den = this->m_rho.state(FieldState::N);
    auto& repo = alphaeff.repo();
    auto& geom_vec = repo.mesh().Geom();

    const int nlevels = repo.num_active_levels();
    for (int lev=0; lev < nlevels; ++lev) {
        const auto& geom = geom_vec[lev];
        const amrex::Real dx = geom.CellSize()[0];
        const amrex::Real dy = geom.CellSize()[1];
        const amrex::Real dz = geom.CellSize()[2];
        const amrex::Real ds = std::cbrt(dx * dy * dz);

        for (amrex::MFIter mfi(alphaeff(lev)); mfi.isValid(); ++mfi) {
            const auto& bx = mfi.tilebox(); 
            const auto& rho_arr = den(lev).const_array(mfi);
            const auto& tke_arr = (*this->m_tke)(lev).array(mfi);
            const auto& alphaeff_arr = alphaeff(lev).array(mfi);
            const auto& tlscale_arr = this->m_turb_lscale(lev).array(mfi);
            
            amrex::ParallelFor(
                bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                        alphaeff_arr(i, j, k) = lam_diff + 
                            rho_arr(i, j, k) * Ce
                            * tlscale_arr(i,j,k) * std::sqrt(tke_arr(i,j,k))
                            * (1.0 + 2.0 * tlscale_arr(i,j,k)/ds);
                    });
        }
    }
    
}

template <typename Transport>
void OneEqKsgsS94<Transport>::update_scalar_diff(Field& scalar_diff, const std::string& name )
{
    BL_PROFILE("amr-wind::" + this->identifier() + "::update_scalar_diff")

    if (name == "tke") {

        const amrex::Real Ce = this->m_Ce;
        auto& den = this->m_rho.state(FieldState::N);
        auto& repo = scalar_diff.repo();
        
        const int nlevels = repo.num_active_levels();
        for (int lev=0; lev < nlevels; ++lev) {
            for (amrex::MFIter mfi(scalar_diff(lev)); mfi.isValid(); ++mfi) {
                const auto& bx = mfi.tilebox(); 
                const auto& rho_arr = den(lev).const_array(mfi);
                const auto& tke_arr = (*this->m_tke)(lev).array(mfi);
                const auto& scalar_diff_arr = scalar_diff(lev).array(mfi);
                const auto& tlscale_arr = this->m_turb_lscale(lev).array(mfi);
                
                amrex::ParallelFor(
                    bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                            scalar_diff_arr(i, j, k) = 2.0 * rho_arr(i, j, k) * Ce
                                * tlscale_arr(i,j,k) * std::sqrt(tke_arr(i,j,k));
                        });
            }
        }
        
    } else {
        std::cerr << "Unknown scalar to compute diffusivity" << std::endl;
    }
}


} // namespace turbulence

INSTANTIATE_TURBULENCE_MODEL(OneEqKsgsM84);
INSTANTIATE_TURBULENCE_MODEL(OneEqKsgsS94);

} // namespace amr_wind
