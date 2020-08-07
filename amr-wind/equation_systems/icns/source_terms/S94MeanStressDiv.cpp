#include "amr-wind/equation_systems/icns/source_terms/S94MeanStressDiv.H"
#include "amr-wind/CFDSim.H"
#include "amr-wind/utilities/tensor_ops.H"
#include "amr-wind/utilities/trig_ops.H"

#include "AMReX_ParmParse.H"

namespace amr_wind {
namespace pde {
namespace icns {

/** Mean Stress Divergence for Sullivan 94 Ksgs turbulence model
 *
 */
S94MeanStressDiv::S94MeanStressDiv(const CFDSim& sim)
    : m_mean_stress_div(sim.repo().get_field("s94_mean_stress_div"))
{
    static_assert(AMREX_SPACEDIM == 3, "S94MeanStressDiv implementation requires 3D domain");
}

S94MeanStressDiv::~S94MeanStressDiv() = default;

void S94MeanStressDiv::operator()(
    const int lev,
    const amrex::MFIter& mfi,
    const amrex::Box& bx,
    const FieldState fstate,
    const amrex::Array4<amrex::Real>& src_term) const
{
    const auto& msd_arr = (this->m_mean_stress_div)(lev).const_array(mfi);

    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
        for (int idim=0; idim  < AMREX_SPACEDIM; idim++)
            src_term(i, j, k, idim) += msd_arr(i,j,k,idim);
    });
}

} // namespace icns
} // namespace pde
} // namespace amr_wind
