#ifndef KSGSM84SRC_H
#define KSGSM84SRC_H

#include "amr-wind/equation_systems/tke/TKESource.H"

namespace amr_wind {
namespace pde {
namespace tke {

/** One-equation Ksgs LES turbulence model
 *   \ingroup turb_model
 *
 *  Reference to Moeng paper
 */
class KsgsM84Src : public TKESource::Register<KsgsM84Src>
{
public:
    static const std::string identifier() { return "KsgsM84Src"; }

    KsgsM84Src(const CFDSim&);

    virtual ~KsgsM84Src();

    virtual void operator()(
        const int lev,
        const amrex::MFIter& mfi,
        const amrex::Box& bx,
        const FieldState fstate,
        const amrex::Array4<amrex::Real>& src_term) const override;

private:

    Field& m_shear_prod;
    Field& m_buoy_prod;
    Field& m_diss;
    
};

} // namespace tke
} // namespace pde
} // namespace amr_wind

#endif /* KSGSM84SRC_H */
