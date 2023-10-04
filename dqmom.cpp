/********************************************************************************************** 
* For parallel processing
**********************************************************************************************/
// #include <omp.h>

/**********************************************************************************************
* For ODE Solve
**********************************************************************************************/
#include <cvode/cvode.h>
#include <sundials/sundials_nvector.h>
#include <sundials/sundials_types.h>
#include <nvector/nvector_serial.h>
#include <sunmatrix/sunmatrix_dense.h>
#include <sunlinsol/sunlinsol_dense.h>
#include <sundials/sundials_context.h>

/**********************************************************************************************
* For Bayesian Inversion
**********************************************************************************************/
// #include <MUQ/Modeling/ModPiece.h>
// #include "MUQ/Modeling/WorkGraph.h"
// #include "MUQ/Modeling/ModGraphPiece.h"
// #include "MUQ/Modeling/SumPiece.h"

// #include "MUQ/Modeling/Distributions/UniformBox.h"
// #include "MUQ/Modeling/Distributions/Gaussian.h"
// #include "MUQ/Modeling/Distributions/Density.h"

// #include "MUQ/SamplingAlgorithms/SamplingProblem.h"
// #include "MUQ/SamplingAlgorithms/SingleChainMCMC.h"
// #include "MUQ/SamplingAlgorithms/MCMCFactory.h"
// #include "MUQ/SamplingAlgorithms/Diagnostics.h"

// #include <boost/property_tree/ptree.hpp>
// #include <boost/math/statistics/univariate_statistics.hpp>

/**********************************************************************************************
* Standard C++ library
**********************************************************************************************/
#include <iostream>
#include <vector>
#include <cmath>
#include <sstream>
#include <random>
#include <fstream>
#include <string>
#include <numeric>
#include <functional>
#include <array>
#include <valarray>
#include <limits>
#include <algorithm>
#include <utility>

/**********************************************************************************************
* Header files specific to this code
**********************************************************************************************/
#include "user_data.hpp"

/**********************************************************************************************
* Helper functions (implemented after main)
**********************************************************************************************/

// Converts number of atoms to a diameter
realtype atoms2diam(const int atoms);

// Computes the matrix in DQMOM needed to evolve quadrature
void computeAmatrix(N_Vector u, void* user_data);

// Computes the agglomeration kernel between two sizes
realtype computeAgglomerationKernel(const realtype x, const realtype y, const std::vector<realtype> & prm);

// Computes the source vector in DQMOM needed to evolve quadrature
void computeSourceVector(N_Vector u, void* user_data);

/**********************************************************************************************
* Functions for ODE integration (implemented after main)
**********************************************************************************************/

// ODE right hand side function
int f(realtype time, N_Vector u, N_Vector du, void* user_data);

/**********************************************************************************************
* Functions for fitting to data
**********************************************************************************************/

// Splits a string based on delimiter
std::vector<std::string> SplitString(std::string str, std::string delimeter);

// Loads the data
MeasurementData readMeasuredData(const std::string filename, const int skip_rows);

std::vector<realtype> trapezoidalRule(const std::vector<realtype> & x, const std::vector<realtype> & y);

// Solves the ODE and then compares to the data
std::pair<realtype, std::vector< std::vector<realtype> >> odeSolveVersusData(const std::valarray<realtype> &  parameters, const MeasurementData & data);

// Class to coordinate with MUQ for Bayesian inversion
// class LogPosterior : public muq::Modeling::ModPiece 
// {
//     public:
//     // Constructor
//     LogPosterior(const int nPrm, const MeasurementData & data);

//     protected:
//     // Evaluates the posterior probability
//     void EvaluateImpl(muq::Modeling::ref_vector<Eigen::VectorXd> const & inputs) override;

//     private:
//     // Compute prior probability
//     realtype LogPrior(muq::Modeling::ref_vector<Eigen::VectorXd> const & inputs);

//     // Compute log likelihood
//     realtype LogLikelihood(muq::Modeling::ref_vector<Eigen::VectorXd> const & inputs);

//     // Measurement data
//     MeasurementData data;
// };


/**********************************************************************************************
* Main program -- Solves the ODE
**********************************************************************************************/
int main(int argc, char** argv){
    int n_chains = 2;
    int n_samples = 1;
    int n_burnin = 0;
    if (argc>1){
        n_chains = atoi(argv[1]);
    }
    if (argc>2){
        n_samples = atoi(argv[2]);
    }
    if (argc>3){
        n_burnin = atoi(argv[3]);
    }

    std::cout << "Using " << n_chains << " chain(s)\n";
    std::cout << "Generating " << n_samples << " sample(s)\n";
    std::cout << "Burning the first " << n_burnin << " samples\n";

    //---------------------------------------------
    // How many parameters in the model
    // Reduction of precursor       --> 1   (in-use)
    // Constant agglomeration       --> 1   (not used)
    // Size-dependent agglomeration --> 4   (in-use)
    // Constant growth              --> 1   (not used)
    // Size-dependent growth        --> 4   (not used)
    // _______________________________________________
    // Total                        --> 5
    //---------------------------------------------
    const int n_prm = 5;
    std::vector< std::string > variable_names = {"Reduction K",
        "Adhesion C1",
        "Adhesion C2",
        "Adhesion P1",
        "Adhesion P2"};

    for (auto s : variable_names){
        std::cout << s << "\n";
    }

        
    return 0;
}


/**********************************************************************************************
* Implementations of helper function
**********************************************************************************************/

realtype atoms2diam(const int atoms)
{
    /*
        input: atoms --> The number of gold atoms in a cluster

        output: diam --> The diameter (in nanometers) of the cluster

        Methodology: Relationship based on https://doi.org/10.1021/jp808967p
                     atoms = 31 * diam^3
                ==>  diam  = (atoms/31)^(1/3)
    */
   return std::pow(1.0*atoms/31.0, 1.0/3.0);
}


void computeAmatrix(N_Vector u, void* user_data)
{
    auto udata = static_cast<UserData*>(user_data);
    auto uarray = N_VGetArrayPointer(u);

    // ---------------------------------------------------------------------------------------------
    // Compute quadrature points
    // ---------------------------------------------------------------------------------------------

    const auto weight_idx = udata->quad_weight_idx;
    const auto wpoint_idx = udata->quad_point_idx;

    std::vector<realtype> quad_points(udata->N); // = (weighted point) / weight

    int n_zeros = 1;
    for (sunindextype i=0; i<udata->N; ++i)
    {
        const auto quad_weight = uarray[ weight_idx[i] ];
        const auto quad_weighted_point = uarray[ wpoint_idx[i] ];
        // std::cout << "compute: " << quad_weighted_point << " / " << quad_weight << " = ";
        // If the weight is zero then the quadrature point is undefined (or infinite)
        // Default to setting to zero
        // But also perturb the zero if multiple occur to prevent singularity in the A matrix
        const realtype pert = 0.01; // FIXME
        if (std::abs(quad_weight) < 1e-8){
            quad_points[i] = n_zeros*pert;
            ++n_zeros;
        }
        else {
            // Division should be fine now
            quad_points[i] = quad_weighted_point / quad_weight;
            // std::cout << "(fixed)";
        }
        // std::cout << quad_points[i] << "\n";
    }

    // ---------------------------------------------------------------------------------------------
    // Check quadrature points and perturb if they are too close to avoid an almost-singular matrix
    // ---------------------------------------------------------------------------------------------

    const realtype tol = 1e-3;
    const realtype pert = 0.01; // FIXME
    for (unsigned int i=0; i<udata->N; ++i){
        for (unsigned int j=i+1; j<udata->N; ++j){
            const auto diff = std::abs( quad_points[i] - quad_points[j]);
            if (diff < tol){
                quad_points[j] *= (1+pert);
            }
        }
    }


    // ---------------------------------------------------------------------------------------------
    // Form the A matrix
    // Each row corresponds to a different Moment
    // Row = [ (1-moment)*quad_points.^moment, moment*quad_points.^(moment-1) ]
    // For moments=0,1,...,2N-1
    // ---------------------------------------------------------------------------------------------

    for (sunindextype r=0;r<2*udata->N;++r){
        for (sunindextype c=0;c<udata->N;++c){
            const auto cidx1 = c;
            const auto cidx2 = c + udata->N;

            // FIXME -- will be a different function for different matrices
            SM_ELEMENT_D(udata->A, r, cidx1) = (1.0 - r) * std::pow( quad_points[c], 1.0 * r );
            SM_ELEMENT_D(udata->A, r, cidx2) = (1.0 * r) * std::pow( quad_points[c], r - 1.0 );
        }
    }
    // SUNDenseMatrix_Print(udata->A, stdout);
}


realtype computeAgglomerationKernel(const realtype x, const realtype y, const std::vector<realtype> & prm)
{
    // Brownian kernel times size-dependent adhesion probability
    // Adhesion probability is what is fit with parameters.
    const realtype brownian = std::pow(x+y, 2) / (x*y);
    
    // Adhesion probability is approximated with a symmetric function
    // Prob = C1*x^P1 + C1*y^P1 + C2*(xy)^P2
    const auto C1 = prm.at(0);
    const auto C2 = prm.at(1);
    const auto P1 = prm.at(2);
    const auto P2 = prm.at(3);

    realtype adhesion = C1 * std::pow(x, P1)
        + C1 * std::pow(y, P1)
        + C2 * std::pow(x*y, P2);

    return adhesion * brownian; 
}


void computeSourceVector(N_Vector u, void* user_data)
{
    auto udata = static_cast<UserData*>(user_data);
    auto uarray = N_VGetArrayPointer(u);
    auto sourcearray = N_VGetArrayPointer(udata->source);

    // ---------------------------------------------------------------------------------------------
    // Source due to particle nucleation
    // Point source nucleation at 1 atom
    // The rate is: nuc_rate = k_reduct * [Precursor] * [Reducer] / diam(1 atom)
    // Need to compute integral nuc_rate * x^M * delta( x - x(1 atom) ) dx
    // Which equals: nuc_rate * x(1 atom)^M
    // where M is the moment
    // ---------------------------------------------------------------------------------------------
    const realtype nuc_rate = udata->reductK
                            * uarray[udata->Aidx]
                            * uarray[udata->Ridx]
                            / atoms2diam(1);
    // std::cout << "nuc rate = " << nuc_rate << std::endl;
    for (sunindextype M=0;M<2*udata->N;++M){
        sourcearray[M] += nuc_rate
            * std::pow( atoms2diam(1), 1.0*M);
    }


    // ---------------------------------------------------------------------------------------------
    // Compute quadrature points
    // ---------------------------------------------------------------------------------------------

    const auto weight_idx = udata->quad_weight_idx;
    const auto wpoint_idx = udata->quad_point_idx;

    std::vector<realtype> quad_points(udata->N); // = (weighted point) / weight
    std::vector<realtype> quad_weights(udata->N);

    for (sunindextype i=0; i<udata->N; ++i)
    {
        quad_weights[i] = uarray[ weight_idx[i] ];
        const auto quad_weighted_point = uarray[ wpoint_idx[i] ];
        // If the weight is zero then the quadrature point is undefined (or infinite)
        // Default to setting to zero
        if (quad_weights[i] == 0.0){
            quad_points[i] = 0.0;
        }
        else {
            // Division should be fine if in this block
            quad_points[i] = quad_weighted_point / quad_weights[i];
        }
    }



    // ---------------------------------------------------------------------------------------------
    // Source due to particle agglomeration
    // Birth:
    //  sum[i]
    //      sum[j]
    //          0.5 * wi * wj * (xi^3 + xj^3)^(moment/3) * kernel(xi, xj)
    //      end
    //  end 
    // Death:
    //  sum[i]
    //      sum[j]
    //          xi^moment * wi * wj * kernel(xi, xj) <-- is there a typo here?? FIXME
    //      end
    //  end   
    // ---------------------------------------------------------------------------------------------
    for (sunindextype M=0;M<2*(udata->N);++M){
        // Moment M
        for (sunindextype i=0; i<(udata->N);++i){
            for (sunindextype j=0; j<(udata->N);++j){
                // Birth
                sourcearray[M] += 0.5 * quad_weights[i] * quad_weights[j]
                    * std::pow( std::pow(quad_points[i], 3.0) + std::pow(quad_points[j], 3.0) ,M/3.0 )
                    * computeAgglomerationKernel(quad_points[i], quad_points[j], udata->agglom_prm);
                // Death
                sourcearray[M] -= std::pow( quad_points[i], 1.0*M) 
                    * quad_weights[i] 
                    * quad_weights[j]
                    * computeAgglomerationKernel(quad_points[i], quad_points[j], udata->agglom_prm);
            }
        }
    }
}


/**********************************************************************************************
* Implementations of ODE integration functions
**********************************************************************************************/

int f(realtype time, N_Vector u, N_Vector du, void* user_data)
{
    auto udata = static_cast<UserData*>(user_data);
    auto uarray = N_VGetArrayPointer(u);
    

    // ---------------------------------------------------------------------------------------------
    // Set previously used vectors and matrices to zero
    // ---------------------------------------------------------------------------------------------
    N_VConst(0.0, du);
    N_VConst(0.0, udata->source);
    N_VConst(0.0, udata->ab_vec);
    SUNMatZero(udata->A);

    auto duarray = N_VGetArrayPointer(du);
    /*
        DQMOM requires computing the rate of change of the quadrature points and weights.
        This is done by solving a linear system of the form
            A * [a b]^T = S
        which is described in: https://doi.org/10.1016/j.jaerosci.2004.07.009
        a --> the rate of change for the weights
        b --> the rate of change for the weighted quadrature points

        Data structures for A, z = [a b]^T, and S are stored in user_data
        The linear solver is also stored in user_data
    */

   computeAmatrix(u, user_data);
   computeSourceVector(u, user_data);
//    N_VPrint_Serial(udata->source);

   SUNLinSolInitialize(udata->lin_solv);
   SUNLinSolSetup(udata->lin_solv, udata->A);
   // Solves A*ab_vec = source for the unknown vector ab_vec
   SUNLinSolSolve(udata->lin_solv, udata->A, udata->ab_vec, udata->source, 1.0e-8);

   // Reduction of precursor
   // Prec + Reduct -->[k] Au1
   duarray[udata->Aidx] = - udata->reductK
                            * uarray[udata->Aidx]
                            * uarray[udata->Ridx];
   duarray[udata->Ridx] = duarray[udata->Aidx];
   // This reaction causes nucleation to occur in the PBE
   // The computeSourceVector function handles this

   // Add [a b] to the right hand side vector
   const auto N = udata->N;
   auto ab_array = N_VGetArrayPointer(udata->ab_vec);
   for (sunindextype i=0; i<N; ++i)
   {
        const auto widx = udata->quad_weight_idx[i];
        const auto ptidx = udata->quad_point_idx[i];
        duarray[widx] = ab_array[i];
        duarray[ptidx] = ab_array[i+N];
   }


    return 0; // FIXME
}

/**********************************************************************************************
* Implementations of inversion functions
**********************************************************************************************/

std::vector<std::string> SplitString(std::string str, std::string delimeter)
{
    std::vector<std::string> splittedStrings = {};
    size_t pos = 0;

    while ((pos = str.find(delimeter)) != std::string::npos)
    {
        std::string token = str.substr(0, pos);
        if (token.length() > 0)
            splittedStrings.push_back(token);
        str.erase(0, pos + delimeter.length());
    }

    if (str.length() > 0)
        splittedStrings.push_back(str);
    return splittedStrings;
}


MeasurementData readMeasuredData(const std::string filename, const int skip_rows)
{
    MeasurementData data;
    std::ifstream infile(filename); // opens the file

    std::string line;
    
    // burn the first set of rows
    for (unsigned int i=0;i<skip_rows;++i){
        std::getline(infile, line);
        // getline reads the next line but we do nothing with it so it "skips" it
    }

    // Assume the first "non-skipped" line provides measurement times
    std::getline(infile, line); // line holds the data times
    auto time_strings = SplitString(line, "\t"); // "\t" means tab delimited
    for (auto t : time_strings){
        data.times.push_back(
            std::stod(t) // converts string to double
        );
    }
    // A density vector for each time
    std::vector< std::vector<realtype> > data_density(data.times.size());

    // Remaining rows are the measurements
    // Format
    // diameter <tab> time1 value <tab> time2 value <tab> ...
    while (std::getline(infile, line)){
        // Split the line
        // if delimiter is a space: " "
        // if delimiter is a tab: "\t"
        auto split_row = SplitString(line, "\t");
        const auto diam = std::stod(split_row[0]);
        // If the diameter is negative then this is noise and ignore it
        // Also ignore diameters greater than 20
        // FIXME
        if (diam > 0.0 && diam < 20.0){
            data.diameters.push_back(diam);
            // Loop through the rest of the row to retrieve densities
            for (unsigned int i=1; i<split_row.size();++i){
                // Density should be non-negative
                const realtype ZERO = 0.0;
                data_density[i-1].push_back(
                    std::max(
                        ZERO,
                        std::stod(split_row[i])
                    )
                );
            }
        }
    }

    data.density = data_density;

    infile.close();
    return data;
}


std::vector<realtype> trapezoidalRule(const std::vector<realtype> & x, const std::vector<realtype> & y)
{
    const unsigned int N = x.size();
    std::vector<realtype> integral(N);
    integral[0] = 0.0;

    for (unsigned int i=1; i<N;++i){
        const auto dx = x[i] - x[i-1];
        integral[i] = integral[i-1] + 0.5 * dx * (y[i] + y[i-1]);
    }
    return integral;
}


std::pair<realtype, std::vector< std::vector<realtype> >> odeSolveVersusData(const std::valarray<realtype> &  parameters, const MeasurementData & data)
{
    // Read the parameters first
    // If any are negative then we can go ahead and skip the simulation
    UserData user_data;

    // Parameters are:
        // FIXME
        // parameters[0] --> reduction rate constant
        // parameters[1--end] --> agglomeration parameters
    user_data.reductK = parameters[0];

    std::vector<realtype> agglom_prm;
    for (unsigned int i=1; i<parameters.size(); ++i){
        agglom_prm.push_back( parameters[i] );
    }
    user_data.agglom_prm = agglom_prm;
    
    if (user_data.reductK < 0.0){
        // Unphysical means we return lowest possible number
        return {std::numeric_limits<realtype>::lowest(), {} };
    } else {
        // SUNDIALS context -- needed internally by SUNDIALS to solve the ODEs
        sundials::Context sunctx;

        // --------------------------------------------------------------
        // Setup user defined data need in ODE solve
        // --------------------------------------------------------------
        

        const sunindextype N = 3; // number of quadrature nodes
        user_data.N = N;

        for (sunindextype i=0; i<N;++i){
            user_data.quad_weight_idx.push_back(i+2);
            user_data.quad_point_idx.push_back(i+N+2);
        }

        N_Vector source = N_VNew_Serial(2*N, sunctx);
        N_Vector ab_vec = N_VNew_Serial(2*N, sunctx);
        user_data.source = source;
        user_data.ab_vec = ab_vec;

        SUNMatrix A = SUNDenseMatrix(2*N, 2*N, sunctx);
        user_data.A = A;

        SUNLinearSolver LS = SUNLinSol_Dense(source, A, sunctx);
        user_data.lin_solv = LS;


        // --------------------------------------------------------------
        // Create vectors and matrices used in ODE solve
        // --------------------------------------------------------------
        N_Vector sol = N_VNew_Serial(2+2*N, sunctx);
        N_Vector ic  = N_VNew_Serial(2+2*N, sunctx);
        N_VConst(0.0, ic);
        NV_Ith_S(ic, user_data.Aidx) = 0.1;
        NV_Ith_S(ic, user_data.Ridx) = 0.3;

        // FIXME
        // weights
        for (unsigned int idx=2; idx<2+N; ++idx){
            NV_Ith_S(ic,idx) = 1e-05; // Add a very small amount of mass to eliminate singularity in beginning
        }

        for (unsigned int idx=2+N; idx<2+N+N; ++idx){
            NV_Ith_S(ic, idx) = 1e-05 * (atoms2diam(1) + idx - (2+N)*0.1); // Put mass near size of a 1 atoms 
        }
        // NV_Ith_S(ic, 2) = 3.2718979324e-01;
        // NV_Ith_S(ic, 3) = 5.8519516564e-01;
        // NV_Ith_S(ic, 4) = 8.7615041119e-02;

        // weighted points
        // NV_Ith_S(ic, 5) = 6.5110313876e-01;
        // NV_Ith_S(ic, 6) = 1.8841204101e+00;
        // NV_Ith_S(ic, 7) = 4.2812102548e-01;

        SUNMatrix template_matrix = SUNDenseMatrix(2+2*N, 2+2*N, sunctx);


        // --------------------------------------------------------------
        // Create linear solver
        // --------------------------------------------------------------
        SUNLinearSolver cvodeLS = SUNLinSol_Dense(sol, template_matrix, sunctx);


        // --------------------------------------------------------------
        // Setup ODE solver (CVODE)
        // --------------------------------------------------------------
        void* cvode_mem = CVodeCreate(CV_BDF, sunctx);
        CVodeSetErrFile(cvode_mem, NULL); // FIXME
        CVodeInit(cvode_mem, f, 0.0, ic);

        CVodeSetUserData(cvode_mem, (void*)&user_data);

        CVodeSetLinearSolver(cvode_mem, cvodeLS, template_matrix);

        CVodeSStolerances(cvode_mem, 1e-3, 1e-8); // FIXME

        CVodeSetMaxNumSteps(cvode_mem, 10000);


        // --------------------------------------------------------------
        // Loop over desired solve times
        // --------------------------------------------------------------

        realtype t = 0.0;
        // FIXME
        // Starting at time 2.0s to avoid nucleation parts (aka assuming pure aggregation)
        // Hence map 2.0s->0.0s and therefore data time->data time - 2.0

        // time 2.0 = data.times[1] so start from dataset 2 and go to end
        realtype loglikeli = 0.0;
        std::vector< std::vector<realtype> > moments012_timei;
        std::pair<realtype, std::vector< std::vector<realtype> >> result;
        bool keep_going = true;
        for (unsigned int dataset=1; dataset<data.times.size();++dataset){
            if (keep_going){
                const auto solve_time = data.times[dataset];
                // std::cout << "Solving to time " << solve_time << "\n";
                auto err = CVode(cvode_mem, solve_time, sol, &t, CV_NORMAL);
                if (err < 0){
                    result = {std::numeric_limits<realtype>::lowest(), {} };
                    keep_going = false;
                }

                // moments -- FIXME
                std::vector<realtype> moments012;
                for (int M=0; M<6; ++M){
                    const realtype mom = NV_Ith_S(sol, 2) * std::pow(NV_Ith_S(sol, 5) / NV_Ith_S(sol, 2), M)
                        + NV_Ith_S(sol, 3) * std::pow(NV_Ith_S(sol, 6) / NV_Ith_S(sol, 3), M)
                        + NV_Ith_S(sol, 4) * std::pow(NV_Ith_S(sol, 7) / NV_Ith_S(sol, 4), M);
                    moments012.push_back(mom);
                }
                moments012_timei.push_back(moments012);

                // Parameters for lognormal distribution
                const realtype mu = std::log( 
                    std::pow(moments012[1], 2) 
                    * (std::pow(moments012[0], -1.5) 
                    * std::pow(moments012[2], -0.5))
                );
                const realtype sigma2 = std::log(
                    moments012[2]
                    * moments012[0]
                    * std::pow(moments012[1], -2)
                );

                auto sim_psd = [&](const realtype x){
                    return std::pow(
                            x * std::sqrt( sigma2 * 2 * M_PI ),
                            -1
                        )
                        * std::exp(
                            - std::pow( std::log(x) - mu, 2)
                            / ( 2 * sigma2)
                        );
                };

                auto sim_cdf = [&](const realtype x){
                    return 0.5
                        * (
                            1
                            + std::erf(
                                (std::log(x) - mu)
                                / std::sqrt( 2 * sigma2 )
                            )
                        );
                };

                // Compute CDF of the data
                const auto cumulative_data = trapezoidalRule(data.diameters, data.density[dataset]);
                const auto norm_factor = cumulative_data.back();
                
                // Compute absolute difference between data and sim CDF
                std::vector<realtype> abs_diff;
                for (unsigned int i=0;i<cumulative_data.size();++i){
                    const auto d = data.diameters[i];
                    const auto sim = sim_cdf(d);
                    const auto data_cdf = cumulative_data[i] / norm_factor;
                    abs_diff.push_back(
                        std::abs(sim - data_cdf)
                    );
                }

                // Integrate the CDF differences to get Wasserstein distance
                const auto wasserstein_vec = trapezoidalRule(data.diameters, abs_diff);
                loglikeli -= wasserstein_vec.back();
            }
        }
        

        // --------------------------------------------------------------
        // Manually delete SUNDIALS objects to clean up memory
        // --------------------------------------------------------------

        N_VDestroy(sol);
        sol = nullptr;
        N_VDestroy(ic);
        ic = nullptr;

        SUNMatDestroy(template_matrix);
        template_matrix = nullptr;

        SUNLinSolFree(cvodeLS);
        cvodeLS = nullptr;

        CVodeFree(&cvode_mem);
        cvode_mem = nullptr;

        if (keep_going){
            return {loglikeli, moments012_timei};
        } else{
            return result;
        }
    }
}


