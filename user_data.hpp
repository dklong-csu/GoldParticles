// SUNDIALS types
#include <sundials/sundials_nvector.h>
#include <sundials/sundials_types.h>
// matrix
// linear solver

#include <vector>

/**********************************************************************************************
* A structure that holds information needed to solve the ODE
**********************************************************************************************/

struct UserData
{
    // What does each index in the ODE vector track
    sunindextype Aidx = 0; // Gold precursor (bonded with a salt)
    sunindextype Ridx = 1; // Reducing agent

    std::vector<sunindextype> quad_weight_idx; // Needs to be defined
    std::vector<sunindextype> quad_point_idx; // Needs to be defined

    // Number of quadrature points
    sunindextype N = 3; // 3 by default, but can be changed

    // For the reduction reaction A + R -> Au
    realtype reductK = 1.0;

    // For approximating the adhesion probability in the agglomeration kernel
    std::vector<realtype> agglom_prm; 

    // For the DQMOM linear solve
    SUNMatrix A = nullptr; // Need to allocate this
    N_Vector source = nullptr; // Need to allocate this
    N_Vector ab_vec = nullptr; // Need to allocate this
    SUNLinearSolver lin_solv = nullptr; // Need to allocate this

    // Destructor --> have to destroy SUNDIALS objects here
    ~UserData();
};


UserData::~UserData()
{
    N_VDestroy(source);
    source = nullptr;
    N_VDestroy(ab_vec);
    ab_vec = nullptr;
    SUNMatDestroy(A);
    A = nullptr;
    SUNLinSolFree(lin_solv);
    lin_solv = nullptr;
}



/**********************************************************************************************
* A structure that holds information about the measurement
**********************************************************************************************/
struct MeasurementData
{
    // Diameters are same for every data set
    std::vector<realtype> diameters;

    // Times data is measured
    std::vector<realtype> times;

    // Density at each time
    std::vector< std::vector<realtype> > density;
};


/**********************************************************************************************
* A structure that holds results of Bayesian inversion
**********************************************************************************************/
struct MCMCResults
{
    // Samples for all chains
    std::vector< std::vector< std::valarray<realtype> > > samples;

    // Posterior values
    std::vector< std::vector<realtype> > posterior_values;
};


