#include <AMReX_Orientation.H>

#include "amr-wind/equation_systems/tke/source_terms/KsgsS94Src.H"
#include "amr-wind/CFDSim.H"
#include "amr-wind/turbulence/TurbulenceModel.H"

namespace amr_wind {
namespace pde {
namespace tke {

KsgsS94Src::KsgsS94Src(const CFDSim& sim)
  : m_shear_prod(sim.repo().get_field("shear_prod")),
    m_buoy_prod(sim.repo().get_field("buoy_prod")),
    m_diss(sim.repo().get_field("sfs_ke_diss"))
{
    AMREX_ALWAYS_ASSERT(sim.turbulence_model().model_name() == "OneEqKsgsS94");
}

KsgsS94Src::~KsgsS94Src() = default;

void KsgsS94Src::operator()(
    const int lev,
    const amrex::MFIter& mfi,
    const amrex::Box& bx,
    const FieldState,
    const amrex::Array4<amrex::Real>& src_term) const
{
    const auto& shear_prod_arr = (this->m_shear_prod)(lev).array(mfi);
    const auto& buoy_prod_arr = (this->m_buoy_prod)(lev).array(mfi);
    const auto& diss_arr = (this->m_diss)(lev).array(mfi);

    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
      src_term(i, j, k) += shear_prod_arr(i,j,k)+ buoy_prod_arr(i,j,k)
          + diss_arr(i,j,k);
    });
    
}

}
}
}
