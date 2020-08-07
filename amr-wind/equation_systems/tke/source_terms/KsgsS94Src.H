#ifndef KSGSS94SRC_H
#define KSGSS94SRC_H

#include "amr-wind/equation_systems/tke/TKESource.H"

namespace amr_wind {
namespace pde {
namespace tke {

/** One-equation Ksgs LES turbulence model
 *   \ingroup turb_model
 *
 *  Reference to Sullivan 1994 paper
 */
class KsgsS94Src : public TKESource::Register<KsgsS94Src>
{
public:
    static const std::string identifier() { return "KsgsS94Src"; }

    KsgsS94Src(const CFDSim&);

    virtual ~KsgsS94Src();

    virtual void operator()(
        const int lev,
        const amrex::MFIter& mfi,
        const amrex::Box& bx,
        const FieldState fstate,
        const amrex::Array4<amrex::Real>& src_term) const override;

private:
    amrex::Real m_Ceps{0.93};
    amrex::Real m_CepsGround{3.9};

    Field& m_turb_lscale;
    Field& m_shear_prod;
    Field& m_buoy_prod;
    Field& m_tke;
    
};
}
}
}

#endif /* KSGSS94SRC_H */
