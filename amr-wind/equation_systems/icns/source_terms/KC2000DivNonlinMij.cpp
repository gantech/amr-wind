#include "amr-wind/equation_systems/icns/source_terms/KC2000DivNonlinMij.H"
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
KC2000DivNonlinMij::KC2000DivNonlinMij(const CFDSim& sim)
    : m_div_nonlin_mij(sim.repo().get_field("kc2000_div_nonlin_mij"))
{
    static_assert(AMREX_SPACEDIM == 3, "KC2000DivNonlinMij implementation requires 3D domain");
}

KC2000DivNonlinMij::~KC2000DivNonlinMij() = default;

void KC2000DivNonlinMij::operator()(
    const int lev,
    const amrex::MFIter& mfi,
    const amrex::Box& bx,
    const FieldState fstate,
    const amrex::Array4<amrex::Real>& src_term) const
{
    const auto& div_mij_arr = m_div_nonlin_mij(lev).const_array(mfi);

    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
        for (int idim=0; idim  < AMREX_SPACEDIM; idim++)
            src_term(i, j, k, idim) += div_mij_arr(i,j,k,idim);
    });
}

} // namespace icns
} // namespace pde
} // namespace amr_wind
