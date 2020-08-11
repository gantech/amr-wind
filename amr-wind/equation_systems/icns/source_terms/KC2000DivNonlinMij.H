#ifndef KC2000DIVNONLINMIJ_H
#define KC2000DIVNONLINMIJ_H

#include "amr-wind/core/FieldRepo.H"
#include "amr-wind/equation_systems/icns/MomentumSource.H"

namespace amr_wind {
namespace pde {
namespace icns {

/** Divergence of nonlinear portion of subfilter stress in Kosovic-Curry 2000 LES model
 *  \ingroup icns_src we_abl
 *
 */
class KC2000DivNonlinMij : public MomentumSource::Register<KC2000DivNonlinMij>
{
public:
    static const std::string identifier() { return "KC2000DivNonlinMij"; }

    KC2000DivNonlinMij(const CFDSim&);

    virtual ~KC2000DivNonlinMij();

    virtual void operator()(
        const int lev,
        const amrex::MFIter& mfi,
        const amrex::Box& bx,
        const FieldState fstate,
        const amrex::Array4<amrex::Real>& src_term) const override;

private:
    const Field& m_div_nonlin_mij;

};

} // namespace icns
} // namespace pde
} // namespace amr_wind

#endif /* KC2000DIVMIJ_H */
