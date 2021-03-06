#ifndef ABL_H
#define ABL_H

#include <memory>

#include "amr-wind/core/Physics.H"
#include "amr-wind/wind_energy/ABLFieldInit.H"
#include "amr-wind/wind_energy/ABLWallFunction.H"
#include "amr-wind/core/SimTime.H"
#include "amr-wind/utilities/FieldPlaneAveraging.H"
#include "amr-wind/wind_energy/ABLStats.H"
/**
 *  \defgroup wind Wind energy modeling
 *  Wind energy modeling
 *
 *  This group documents all the wind-energy related physics models available in
 *  AMR-Wind
 *
 *  \ingroup physics
 */

/**
 *  \defgroup we_abl ABL
 *  Atmospheric boundary layer (ABL) modeling
 *
 *  \ingroup wind
 */

class incflo;

namespace amr_wind {

namespace pde {
namespace icns {
class ABLForcing;
}
}

/** Atmospheric Boundary Layer physics
 *  \ingroup we_abl
 *
 *  This class implements the necessary code to simulation atmospheric boundary
 *  layers with AMR-Wind. It reads the namespace `ABL` from the input file to
 *  determine how the initial conditions are generated. At each timestep it
 *  performs plane averages to collect statistics as well as set the necessary
 *  data to be used by wall models as well as any ABLForcing terms that might be
 *  active.
 *
 *  \sa ABLFieldInit, ABLWallFunction, FieldPlaneAveraging
 */
class ABL : public Physics::Register<ABL>
{
public:
    static const std::string identifier() { return "ABL"; }

    ABL(CFDSim& sim);

    virtual ~ABL();

    const ABLWallFunction& abl_wall_function() const
    { return m_abl_wall_func; }

    void post_init_actions() override;

    void post_regrid_actions() override {}

    void initialize_fields(
        int level,
        const amrex::Geometry& geom) override;

    void pre_advance_work() override;

    void post_advance_work() override;

    void register_forcing_term(pde::icns::ABLForcing* forcing) const
    {
        m_abl_forcing = forcing;
    }
    
private:
    const CFDSim& m_sim;

    Field& m_velocity;
    Field& m_mueff;
    Field& m_density;
    Field* m_temperature{nullptr};
    Field* m_tke{nullptr};

    ABLWallFunction m_abl_wall_func;

    //! ABL field initializer instance
    std::unique_ptr<ABLFieldInit> m_field_init;

    //! Reference to ABL forcing term if present
    mutable pde::icns::ABLForcing* m_abl_forcing{nullptr};

    //! ABL integrated statistics object
    std::unique_ptr<ABLStats> m_stats;
    
};

}

#endif /* ABL_H */
