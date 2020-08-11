#ifndef ONEEQKSGSI_H
#define ONEEQKSGS_H

#include "amr-wind/turbulence/LES/OneEqKsgs.H"
#include "amr-wind/equation_systems/tke/TKE.H"

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

}
    
template <typename Transport>
OneEqKsgs<Transport>::~OneEqKsgs() = default;
 
}
}



#endif /* ONEEQKSGSI_H */
