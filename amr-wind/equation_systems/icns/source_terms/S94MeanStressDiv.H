#ifndef S94MEANSTRESSDIVFORCING_H
#define S94MEANSTRESSDIVFORCING_H

#include "amr-wind/core/FieldRepo.H"
#include "amr-wind/equation_systems/icns/MomentumSource.H"

namespace amr_wind {
namespace pde {
namespace icns {

/** Account for Mean Stress Divergence in Sullivan 94 turbulence model
 *  \ingroup icns_src we_abl
 *
 */
class S94MeanStressDiv : public MomentumSource::Register<S94MeanStressDiv>
{
public:
    static const std::string identifier() { return "S94MeanStressDiv"; }

    S94MeanStressDiv(const CFDSim&);

    virtual ~S94MeanStressDiv();

    virtual void operator()(
        const int lev,
        const amrex::MFIter& mfi,
        const amrex::Box& bx,
        const FieldState fstate,
        const amrex::Array4<amrex::Real>& src_term) const override;

private:
    const Field& m_mean_stress_div;

};

} // namespace icns
} // namespace pde
} // namespace amr_wind

#endif /* S94MEANSTRESSDIVFORCING_H */
