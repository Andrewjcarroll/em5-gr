#include "rhs.h"

#include "debugger_tools.h"
#include "hadrhs.h"
#include "parameters.h"
#include "solver_main.h"

#define PI 3.14159265358979323846

// #define ROOTp () /* ... true iff rank == 0 */
// static void wait_for_debugger() {
//     if (getenv(" TJF_MPI_DEBUG ") != NULL && ROOTp()) {
//         volatile int i = 0;
//         fprintf(stderr, " pid % ld waiting for debugger \ n ",
//         (long)getpid()); while (i == 0) { /* change ’i ’ in the debugger */
//         }
//     }
//     MPI_Barrier(MPI_COMM_WORLD);
// }

// EWH uncomment the below and should get lots more information
// #define SOLVER_DEBUG_RHS_EQNS

using namespace std;
using namespace dsolve;

void solverRHS(double **uzipVarsRHS, double **uZipVars,
               const ot::Block *blkList, unsigned int numBlocks) {
    unsigned int offset;
    double ptmin[3], ptmax[3];
    unsigned int sz[3];
    unsigned int bflag;
    double dx, dy, dz;
    double kappa_1 = 0.1 ; 
    double kappa_2 = 0.1 ; 
    const Point pt_min(dsolve::SOLVER_COMPD_MIN[0], dsolve::SOLVER_COMPD_MIN[1],
                       dsolve::SOLVER_COMPD_MIN[2]);
    const Point pt_max(dsolve::SOLVER_COMPD_MAX[0], dsolve::SOLVER_COMPD_MAX[1],
                       dsolve::SOLVER_COMPD_MAX[2]);
    const unsigned int PW = dsolve::SOLVER_PADDING_WIDTH;

#ifdef SOLVER_ENABLE_CUDA
    cuda::SOLVERComputeParams solverParams;

    dim3 threadBlock(16, 16, 1);
    cuda::computeRHS(uzipVarsRHS, (const double **)uZipVars, blkList, numBlocks,
                     (const cuda::SOLVERComputeParams *)&solverParams,
                     threadBlock, pt_min, pt_max, 1);
#else

    for (unsigned int blk = 0; blk < numBlocks; blk++) {
        offset = blkList[blk].getOffset();
        sz[0] = blkList[blk].getAllocationSzX();
        sz[1] = blkList[blk].getAllocationSzY();
        sz[2] = blkList[blk].getAllocationSzZ();

        bflag = blkList[blk].getBlkNodeFlag();

        dx = blkList[blk].computeDx(pt_min, pt_max);
        dy = blkList[blk].computeDy(pt_min, pt_max);
        dz = blkList[blk].computeDz(pt_min, pt_max);

        ptmin[0] = GRIDX_TO_X(blkList[blk].getBlockNode().minX()) - PW * dx;
        ptmin[1] = GRIDY_TO_Y(blkList[blk].getBlockNode().minY()) - PW * dy;
        ptmin[2] = GRIDZ_TO_Z(blkList[blk].getBlockNode().minZ()) - PW * dz;

        ptmax[0] = GRIDX_TO_X(blkList[blk].getBlockNode().maxX()) + PW * dx;
        ptmax[1] = GRIDY_TO_Y(blkList[blk].getBlockNode().maxY()) + PW * dy;
        ptmax[2] = GRIDZ_TO_Z(blkList[blk].getBlockNode().maxZ()) + PW * dz;

#ifdef EM5_ENABLE_COMPACT_DERIVS
        solverrhs_compact_derivs(uzipVarsRHS, uZipVars, offset, ptmin, ptmax,
                                 sz, bflag);
#else
        solverrhs(uzipVarsRHS, (const double **)uZipVars, offset, ptmin, ptmax,
                  sz, bflag);
#endif
    }
#endif
}

// NOTE: this is the solverRHS function that has CONST in uZipVars, for legacy
// reasons and does not touch CFD
void solverRHS(double **uzipVarsRHS, const double **uZipVars,
               const ot::Block *blkList, unsigned int numBlocks) {
    unsigned int offset;
    double ptmin[3], ptmax[3];
    unsigned int sz[3];
    unsigned int bflag;
    double dx, dy, dz;
    const Point pt_min(dsolve::SOLVER_COMPD_MIN[0], dsolve::SOLVER_COMPD_MIN[1],
                       dsolve::SOLVER_COMPD_MIN[2]);
    const Point pt_max(dsolve::SOLVER_COMPD_MAX[0], dsolve::SOLVER_COMPD_MAX[1],
                       dsolve::SOLVER_COMPD_MAX[2]);
    const unsigned int PW = dsolve::SOLVER_PADDING_WIDTH;

#ifdef SOLVER_ENABLE_CUDA
    cuda::SOLVERComputeParams solverParams;

    dim3 threadBlock(16, 16, 1);
    cuda::computeRHS(uzipVarsRHS, (const double **)uZipVars, blkList, numBlocks,
                     (const cuda::SOLVERComputeParams *)&solverParams,
                     threadBlock, pt_min, pt_max, 1);
#else

    for (unsigned int blk = 0; blk < numBlocks; blk++) {
        offset = blkList[blk].getOffset();
        sz[0] = blkList[blk].getAllocationSzX();
        sz[1] = blkList[blk].getAllocationSzY();
        sz[2] = blkList[blk].getAllocationSzZ();

        bflag = blkList[blk].getBlkNodeFlag();

        dx = blkList[blk].computeDx(pt_min, pt_max);
        dy = blkList[blk].computeDy(pt_min, pt_max);
        dz = blkList[blk].computeDz(pt_min, pt_max);

        ptmin[0] = GRIDX_TO_X(blkList[blk].getBlockNode().minX()) - PW * dx;
        ptmin[1] = GRIDY_TO_Y(blkList[blk].getBlockNode().minY()) - PW * dy;
        ptmin[2] = GRIDZ_TO_Z(blkList[blk].getBlockNode().minZ()) - PW * dz;

        ptmax[0] = GRIDX_TO_X(blkList[blk].getBlockNode().maxX()) + PW * dx;
        ptmax[1] = GRIDY_TO_Y(blkList[blk].getBlockNode().maxY()) + PW * dy;
        ptmax[2] = GRIDZ_TO_Z(blkList[blk].getBlockNode().maxZ()) + PW * dz;

        solverrhs(uzipVarsRHS, (const double **)uZipVars, offset, ptmin, ptmax,
                  sz, bflag);
    }
#endif
}

#if 0
template <typename T>
void printRHSVarStats(T **variables, unsigned int n, const unsigned int offset,
                      const unsigned int bflag, const unsigned int PW,
                      const unsigned int *sz, std::string message = "",
                      std::string var_suffix = "_RHS") {
    std::cout << "    " << message << std::endl;
    if (bflag != 0) {
        std::cout << "        BOUNDARY FLAG DETECTED, WILL NOT INCLUDE PADDING "
                     "BEYOND BOUNDARY, BFLAG: "
                  << bflag << std::endl;
    }

    T l_min, l_max, l2_norm, l_avg;

    for (const auto tmp_varID : dsolve::SOLVER_VAR_ITERABLE_LIST) {
        if (bflag != 0) {
            l_min =
                vecMinNoBoundary(&variables[tmp_varID][offset], sz, bflag, PW);
            l_max =
                vecMaxNoBoundary(&variables[tmp_varID][offset], sz, bflag, PW);
            l2_norm =
                normL2NoBoundary(&variables[tmp_varID][offset], sz, bflag, PW);
            l_avg =
                vecMeanNoBoundary(&variables[tmp_varID][offset], sz, bflag, PW);
        } else {
            l_min = vecMin(&variables[tmp_varID][offset], n);
            l_max = vecMax(&variables[tmp_varID][offset], n);
            l2_norm = normL2(&variables[tmp_varID][offset], n);
            l_avg = vecMean(&variables[tmp_varID][offset], n);
        }

        std::cout << "        ||VAR::" << dsolve::SOLVER_VAR_NAMES[tmp_varID]
                  << var_suffix << "|| (min, max) : (" << l_min << ", " << l_max
                  << " ) - (l2) : " << l2_norm << " - (mean) : " << l_avg
                  << std::endl;
    }
}

template <typename T>
void printRHSVarStats(const T **variables, unsigned int n,
                      const unsigned int offset, const unsigned int bflag,
                      const unsigned int PW, const unsigned int *sz,
                      std::string message = "",
                      std::string var_suffix = "_RHS") {
    std::cout << "    " << message << std::endl;
    if (bflag != 0) {
        std::cout << "        BOUNDARY FLAG DETECTED, WILL NOT INCLUDE PADDING "
                     "BEYOND BOUNDARY, BFLAG: "
                  << bflag << std::endl;
    }

    T l_min, l_max, l2_norm, l_avg;

    for (const auto tmp_varID : dsolve::SOLVER_VAR_ITERABLE_LIST) {
        if (bflag != 0) {
            l_min =
                vecMinNoBoundary(&variables[tmp_varID][offset], sz, bflag, PW);
            l_max =
                vecMaxNoBoundary(&variables[tmp_varID][offset], sz, bflag, PW);
            l2_norm =
                normL2NoBoundary(&variables[tmp_varID][offset], sz, bflag, PW);
            l_avg =
                vecMeanNoBoundary(&variables[tmp_varID][offset], sz, bflag, PW);
        } else {
            l_min = vecMin(&variables[tmp_varID][offset], n);
            l_max = vecMax(&variables[tmp_varID][offset], n);
            l2_norm = normL2(&variables[tmp_varID][offset], n);
            l_avg = vecMean(&variables[tmp_varID][offset], n);
        }

        std::cout << "        ||VAR::" << dsolve::SOLVER_VAR_NAMES[tmp_varID]
                  << var_suffix << "|| (min, max) : (" << l_min << ", " << l_max
                  << " ) - (l2) : " << l2_norm << " - (mean) : " << l_avg
                  << std::endl;
    }
}
#endif

/*----------------------------------------------------------------------;
 *
 * vector form of RHS
 *
 *----------------------------------------------------------------------*/
void solverrhs(double **unzipVarsRHS, const double **uZipVars,
               const unsigned int &offset, const double *pmin,
               const double *pmax, const unsigned int *sz,
               const unsigned int &bflag) {
    // std::cout << "Entering the RHS computation function..." << std::endl;

    // wait_for_debugger();

    // std::cout << "Boundary Flag: " << bflag << std::endl;
    // clang-format off
    /*[[[cog
    import cog
    import sys
    import os
    import importlib.util
    import dendrosym

    cog.outl("// clang-format on")

    # get the current working directory, should be root of project
    current_path = os.getcwd()
    output_path = os.path.join(current_path, "gencode")

    # the following lines will import any module directly from
    spec = importlib.util.spec_from_file_location("dendroconf", CONFIG_FILE_PATH)
    dendroconf = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = dendroconf
    spec.loader.exec_module(dendroconf)

    cog.outl("// EVOLUTION VARIABLE EXTRACTION NOT RHS")
    cog.outl(
        dendroconf.dendroConfigs.generate_variable_extraction("evolution", use_const=True)
    )

    cog.outl("// EVOLUTION VARIABLE EXTRACTION RHS")
    cog.outl(
        dendroconf.dendroConfigs.generate_rhs_var_extraction(
            "evolution", zip_var_name="unzipVarsRHS"
        )
    )


    ]]]*/
 
    //[[[end]]]

    //EVOLUTION VARIABLE EXTRACTION NOT RHS -AJC
    const double *E0 = &uZipVars[VAR::U_E0][offset];
    const double *E1 = &uZipVars[VAR::U_E1][offset];
    const double *E2 = &uZipVars[VAR::U_E2][offset];
    const double *A0 = &uZipVars[VAR::U_A0][offset];
    const double *A1 = &uZipVars[VAR::U_A1][offset];
    const double *A2 = &uZipVars[VAR::U_A2][offset];
    const double *Gamma = &uZipVars[VAR::U_GAMMA][offset];
    const double *psi = &uZipVars[VAR::U_PSI][offset];
    const double *phi = &uZipVars[VAR::U_PHI][offset];

    //EVOLUTION VARIABLE EXTRACTION RHS -AJC
    double *E_rhs0 = &unzipVarsRHS[VAR::U_E0][offset];
    double *E_rhs1 = &unzipVarsRHS[VAR::U_E1][offset];
    double *E_rhs2 = &unzipVarsRHS[VAR::U_E2][offset];
    double *A_rhs0 = &unzipVarsRHS[VAR::U_A0][offset];
    double *A_rhs1 = &unzipVarsRHS[VAR::U_A1][offset];
    double *A_rhs2 = &unzipVarsRHS[VAR::U_A2][offset];
    double *Gamma_rhs = &unzipVarsRHS[VAR::U_GAMMA][offset];
    double *psi_rhs = &unzipVarsRHS[VAR::U_PSI][offset];
    double *phi_rhs = &unzipVarsRHS[VAR::U_PHI][offset];
    mem::memory_pool<double> *__mem_pool = &SOLVER_MEM_POOL;

    const unsigned int nx = sz[0];
    const unsigned int ny = sz[1];
    const unsigned int nz = sz[2];

    const double hx = (pmax[0] - pmin[0]) / (nx - 1);
    const double hy = (pmax[1] - pmin[1]) / (ny - 1);
    const double hz = (pmax[2] - pmin[2]) / (nz - 1);

    // clang-format off
    /*[[[cog
    cog.outl('// clang-format on')
    cog.outl("// PARAMETER EXTRACTION FOR EVOLUTION")

    cog.outl(dendroconf.dendroConfigs.gen_parameter_code("evolution"))

    ]]]*/
    // clang-format on

    //[[[end]]]

    int idx[3];
    const unsigned int PW = dsolve::SOLVER_PADDING_WIDTH;
    unsigned int n = sz[0] * sz[1] * sz[2];

    dsolve::timer::t_deriv.start();

    const unsigned int BLK_SZ = n;
    const unsigned int bytes = n * sizeof(double);
    // get derivative workspace
    double *const deriv_base = dsolve::SOLVER_DERIV_WORKSPACE;

    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Create the necessary pre-derivatives
    // clang-format off
    /*[[[cog
    cog.outl('// clang-format on')

    cog.outl("//GENERATED ADVANCED DERIVATIVE EQUATIONS")

    print("Now generating advanced derivatves", file=sys.stderr)
    
    # note that we need to store the deallocation string as well for later down the line!
    (intermediate_grad_str, 
     deallocate_intermediate_grad_str) = dendroconf.dendroConfigs.generate_pre_necessary_derivatives(
         "evolution", dtype="double", include_byte_declaration=False
     )

    print("Finished generating advanced derivatves", file=sys.stderr)

    intermediate_filename = "solver_rhs_intermediate_grad.cpp.inc"

    with open(os.path.join(output_path, intermediate_filename), "w") as f:
        f.write(intermediate_grad_str)

    print("Saved them to file", file=sys.stderr)

    cog.outl(f'#include "../gencode/{intermediate_filename}"')

    ]]]*/
    // clang-format on

    //[[[end]]]

    // create the files that have the derivative memory allocations and
    // calculations
    // clang-format off
    /*[[[cog
    cog.outl('// clang-format on')

    deriv_alloc, deriv_calc, deriv_dealloc = dendroconf.dendroConfigs.generate_deriv_allocation_and_calc("evolution", include_byte_declaration=False)

    print("Generated derivative allocation, calculation, and deallocation code for Evolution", file=sys.stderr)

    alloc_filename = "solver_rhs_deriv_memalloc.cpp.inc"

    with open(os.path.join(output_path, alloc_filename), "w") as f:
        f.write(deriv_alloc)
    
    cog.outl(f'#include "../gencode/{alloc_filename}"')

    calc_filename = "solver_rhs_deriv_calc.cpp.inc"

    with open(os.path.join(output_path, calc_filename), "w") as f:
        f.write(deriv_calc)
    
    cog.outl(f'#include "../gencode/{calc_filename}"')

    dealloc_filename = "solver_rhs_deriv_memdealloc.cpp.inc"

    with open(os.path.join(output_path, dealloc_filename), "w") as f:
        f.write(deriv_dealloc)

    
    ]]]*/
#include "../gencode/solver_rhs_deriv_memalloc.cpp.inc"
#include "../gencode/solver_rhs_deriv_calc.cpp.inc"
    // clang-format on
    //[[[end]]]

    dsolve::timer::t_deriv.stop();

    // TODO: is this even necessary?
    double *rho_e = __mem_pool->allocate(n);

    double *J0 = __mem_pool->allocate(n);
    double *J1 = __mem_pool->allocate(n);
    double *J2 = __mem_pool->allocate(n);

    for (unsigned int m = 0; m < n; m++) {
        rho_e[m] = 0.0;

        J0[m] = 0.0;
        J1[m] = 0.0;
        J2[m] = 0.0;
    }

    // loop dep. removed allowing compiler to optmize for vectorization.
    // cout << "begin loop" << endl;
    dsolve::timer::t_rhs.start();
    for (unsigned int k = PW; k < nz - PW; k++) {
        for (unsigned int j = PW; j < ny - PW; j++) {
#ifdef SOLVER_ENABLE_AVX
#ifdef __INTEL_COMPILER
#pragma vector vectorlength(__RHS_AVX_SIMD_LEN__) vecremainder
#pragma ivdep
#endif
#endif
            for (unsigned int i = PW; i < nx - PW; i++) {
                const double x = pmin[0] + i * hx;
                const double y = pmin[1] + j * hy;
                const double z = pmin[2] + k * hz;

                double kappa_1 = 0.1 ; 
                double kappa_2 = 0.1 ;   

                const unsigned int pp = i + nx * (j + ny * k);
                const double r_coord = sqrt(x * x + y * y + z * z);

                // clang-format off
                /*[[[cog
                cog.outl('// clang-format on')

                evolution_rhs_code = dendroconf.dendroConfigs.generate_rhs_code("evolution")
                evolution_filename = "solver_rhs_eqns.cpp.inc"

                with open(os.path.join(output_path, evolution_filename), "w") as f:
                    f.write(evolution_rhs_code)
                
                cog.outl(f'#include "../gencode/{evolution_filename}"')
                
                ]]]*/

#include "../gencode/solver_rhs_eqns.cpp.inc"

                //[[[end]]]

            }
        }
    }
    dsolve::timer::t_rhs.stop();

    // Deallocate the pre-derivatives
    // TODO: is this the best place to put this? or should it reside at the end
    // with the rest of the freeing?
    // clang-format off
    /*[[[cog
    cog.outl('// clang-format on')

    cog.outl("//GENERATED DEALLOCATION OF INTERMEDIATE GRAD CALCULATIONS")

    cog.outl(deallocate_intermediate_grad_str)

    ]]]*/
    // clang-format on
    // GENERATED DEALLOCATION OF INTERMEDIATE GRAD CALCULATIONS

    //[[[end]]]

    if (bflag != 0) {
        dsolve::timer::t_bdyc.start();

        // asymptotic_and_falloff_bcs(E_rhs0, E0, grad_0_E0, grad_1_E0, grad_2_E0,
        //                            pmin, pmax, 2.0, 0.0, sz, bflag);
        // asymptotic_and_falloff_bcs(E_rhs1, E1, grad_0_E1, grad_1_E1, grad_2_E1,
        //                            pmin, pmax, 2.0, 0.0, sz, bflag);
        // asymptotic_and_falloff_bcs(E_rhs2, E2, grad_0_E2, grad_1_E2, grad_2_E2,
        //                            pmin, pmax, 2.0, 0.0, sz, bflag);

        // asymptotic_and_falloff_bcs(A_rhs0, A0, grad_0_A0, grad_1_A0, grad_2_A0,
        //                            pmin, pmax, 1.0, 0.0, sz, bflag);
        // asymptotic_and_falloff_bcs(A_rhs1, A1, grad_0_A1, grad_1_A1, grad_2_A1,
        //                            pmin, pmax, 1.0, 0.0, sz, bflag);
        // asymptotic_and_falloff_bcs(A_rhs2, A2, grad_0_A2, grad_1_A2, grad_2_A2,
        //                            pmin, pmax, 1.0, 0.0, sz, bflag);

        // asymptotic_and_falloff_bcs(Gamma_rhs, Gamma, grad_0_Gamma, grad_1_Gamma, grad_2_Gamma,
        //                            pmin, pmax, 1.0, 0.0, sz, bflag);
        // asymptotic_and_falloff_bcs(psi_rhs, psi, grad_0_psi, grad_1_psi, grad_2_psi,
        //                            pmin, pmax, 1.0, 0.0, sz, bflag);

        asymptotic_and_falloff_bcs(Gamma_rhs, Gamma, grad_0_Gamma, grad_1_Gamma, grad_2_Gamma, 
        pmin, pmax, 2.0, 0.0, sz, bflag);

        asymptotic_and_falloff_bcs(psi_rhs, psi, grad_0_psi, grad_1_psi, grad_2_psi, pmin, pmax,
        1.0, 0.0, sz, bflag);

        asymptotic_and_falloff_bcs(phi_rhs, phi, grad_0_phi, grad_1_phi, grad_2_phi, pmin, pmax,
        1.0, 0.0, sz, bflag);

        asymptotic_and_falloff_bcs(E_rhs0, E0, grad_0_E0, grad_1_E0, grad_2_E0, pmin, pmax,
        2.0, 0.0, sz, bflag);
        asymptotic_and_falloff_bcs(E_rhs1, E1, grad_0_E1, grad_1_E1, grad_2_E1, pmin, pmax,
        2.0, 0.0, sz, bflag);
        asymptotic_and_falloff_bcs(E_rhs2, E2, grad_0_E2, grad_1_E2, grad_2_E2, pmin, pmax,
        2.0, 0.0, sz, bflag);

        asymptotic_and_falloff_bcs(A_rhs0, A0, grad_0_A0, grad_1_A0, grad_2_A0, pmin, pmax,
        1.0, 0.0, sz, bflag);
        asymptotic_and_falloff_bcs(A_rhs1, A1, grad_0_A1, grad_1_A1, grad_2_A1, pmin, pmax,
        1.0, 0.0, sz, bflag);
        asymptotic_and_falloff_bcs(A_rhs2, A2, grad_0_A2, grad_1_A2, grad_2_A2, pmin, pmax,
        1.0, 0.0, sz, bflag);

        //[[[end]]]

        dsolve::timer::t_bdyc.stop();
    }

    dsolve::timer::t_deriv.start();
    // TODO: include more types of build options

#include "../gencode/solver_rhs_ko_deriv_calc.cpp.inc"
    dsolve::timer::t_deriv.stop();

    dsolve::timer::t_rhs.start();

    const double sigma = KO_DISS_SIGMA;

    for (unsigned int k = PW; k < nz - PW; k++) {
        for (unsigned int j = PW; j < ny - PW; j++) {
#ifdef SOLVER_ENABLE_AVX
#ifdef __INTEL_COMPILER
#pragma vector vectorlength(__RHS_AVX_SIMD_LEN__) vecremainder
#pragma ivdep
#endif
#endif
            for (unsigned int i = PW; i < nx - PW; i++) {
                const unsigned int pp = i + nx * (j + ny * k);
                // Added KO DISSIPATION CALCULATIONS FROM EM5 CODE -AJC
            Gamma_rhs[pp] += sigma*(grad_0_Gamma[pp]+grad_1_Gamma[pp]+grad_2_Gamma[pp]);
            
            psi_rhs[pp] += sigma*(grad_0_psi[pp]+grad_1_psi[pp]+grad_2_psi[pp]);

            phi_rhs[pp] += sigma*(grad_0_phi[pp]+grad_1_phi[pp]+grad_2_phi[pp]);
            
            E_rhs0[pp] += sigma*(grad_0_E0[pp] + grad_1_E0[pp] + grad_2_E0[pp]);
            E_rhs1[pp] += sigma*(grad_0_E1[pp] + grad_1_E1[pp] + grad_2_E1[pp]);
            E_rhs2[pp] += sigma*(grad_0_E2[pp] + grad_1_E2[pp] + grad_2_E2[pp]);
            
            A_rhs0[pp] += sigma*(grad_0_A0[pp] + grad_1_A0[pp] + grad_2_A0[pp]);
            A_rhs1[pp] += sigma*(grad_0_A1[pp] + grad_1_A1[pp] + grad_2_A1[pp]);
            A_rhs2[pp] += sigma*(grad_0_A2[pp] + grad_1_A2[pp] + grad_2_A2[pp]);
                // clang-format off
                /*[[[cog
                cog.outl('// clang-format on')

                cog.outl("// GENERATED KO DISSIPATION CALCULATIONS")
                cog.outl(dendroconf.dendroConfigs.generate_ko_calculations("evolution"))

                ]]]*/
                // clang-format on

                //[[[end]]]
            }
        }
    }

    dsolve::timer::t_rhs.stop();

    dsolve::timer::t_deriv.start();
    // clang-format off
    /*[[[cog
    cog.outl('// clang-format on')

    cog.outl(f'#include "../gencode/{dealloc_filename}"')

    ]]]*/
    // clang-format on

    //[[[end]]]
    __mem_pool->free(J0);
    __mem_pool->free(J1);
    __mem_pool->free(J2);

    __mem_pool->free(rho_e);
    dsolve::timer::t_deriv.stop();
}

void solverrhs_compact_derivs(double **unzipVarsRHS, double **uZipVars,
                              const unsigned int &offset, const double *pmin,
                              const double *pmax, const unsigned int *sz,
                              const unsigned int &bflag) {
    // NOTE: this has been cleaned up slightly to remove the code generation.
    // if the function above changes, be sure to reflect the changes here
    //

    // EVOLUTION VARIABLE EXTRACTION NOT RHS -AJC
    double *E0 = &uZipVars[VAR::U_E0][offset];
    double *E1 = &uZipVars[VAR::U_E1][offset];
    double *E2 = &uZipVars[VAR::U_E2][offset];
    double *A0 = &uZipVars[VAR::U_A0][offset];
    double *A1 = &uZipVars[VAR::U_A1][offset];
    double *A2 = &uZipVars[VAR::U_A2][offset];
    double *Gamma = &uZipVars[VAR::U_GAMMA][offset];
    double *psi = &uZipVars[VAR::U_PSI][offset];
     double *phi = &uZipVars[VAR::U_PHI][offset];
    // EVOLUTION VARIABLE EXTRACTION RHS -AJC
    double *E_rhs0 = &unzipVarsRHS[VAR::U_E0][offset];
    double *E_rhs1 = &unzipVarsRHS[VAR::U_E1][offset];
    double *E_rhs2 = &unzipVarsRHS[VAR::U_E2][offset];
    double *A_rhs0 = &unzipVarsRHS[VAR::U_A0][offset];
    double *A_rhs1 = &unzipVarsRHS[VAR::U_A1][offset];
    double *A_rhs2 = &unzipVarsRHS[VAR::U_A2][offset];
    double *Gamma_rhs = &unzipVarsRHS[VAR::U_GAMMA][offset];
    double *psi_rhs = &unzipVarsRHS[VAR::U_PSI][offset];
    double *phi_rhs = &unzipVarsRHS[VAR::U_PHI][offset];


    mem::memory_pool<double> *__mem_pool = &SOLVER_MEM_POOL;

    const unsigned int nx = sz[0];
    const unsigned int ny = sz[1];
    const unsigned int nz = sz[2];

    const double hx = (pmax[0] - pmin[0]) / (nx - 1);
    const double hy = (pmax[1] - pmin[1]) / (ny - 1);
    const double hz = (pmax[2] - pmin[2]) / (nz - 1);

    int idx[3];
    const unsigned int PW = dsolve::SOLVER_PADDING_WIDTH;
    unsigned int n = sz[0] * sz[1] * sz[2];

    dsolve::timer::t_deriv.start();

    const unsigned int BLK_SZ = n;
    const unsigned int bytes = n * sizeof(double);
    // get derivative workspace
    double *const deriv_base = dsolve::SOLVER_DERIV_WORKSPACE;

    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

#include "../gencode/solver_rhs_deriv_memalloc.cpp.inc"

    // store a pointer to the variables that will be sent into our derivatives
    // if we're not filtering there's no reason to change it
    double *E0_cpy = (double *)E0;
    double *E1_cpy = (double *)E1;
    double *E2_cpy = (double *)E2;
    double *A0_cpy = (double *)A0;
    double *A1_cpy = (double *)A1;
    double *A2_cpy = (double *)A2;
    double *Gamma_cpy = (double *)Gamma;
    double *psi_cpy = (double *)psi;
    double *phi_cpy = (double *)phi;

    // make sure we only trigger this filtering if it's a filter designed for it
    if (SOLVER_DERIVS->do_filter_before()) {
        // for each of the variables, we'll copy it over to the memory stored
        // for it in the copy then it will be filtered. The filtered variables
        // will then feed ONLY into the derivatives. We might want to use the
        // "cpy" version for the RHS computations eventually to see if it helps,
        // but at the very least the derivatives will get the filtered version.
        E0_cpy = E_rhs0;
        E1_cpy = E_rhs1;
        E2_cpy = E_rhs2;
        A0_cpy = A_rhs0;
        A1_cpy = A_rhs1;
        A2_cpy = A_rhs2;
        Gamma_cpy = Gamma_rhs;
        psi_cpy = psi_rhs;
        phi_cpy = phi_rhs;


        // NOTE: the filter function will check if input and output are the same
        // pointer, if they are it'll do the filtering in place, otherwise it
        // will copy the output over and *then* apply to the copy
        SOLVER_DERIVS->filter(E0, E0_cpy, nullptr, nullptr, nullptr, hx, hy, hz,
                              1.0, sz, bflag);
        SOLVER_DERIVS->filter(E1, E1_cpy, nullptr, nullptr, nullptr, hx, hy, hz,
                              1.0, sz, bflag);
        SOLVER_DERIVS->filter(E2, E2_cpy, nullptr, nullptr, nullptr, hx, hy, hz,
                              1.0, sz, bflag);
        SOLVER_DERIVS->filter(A0, A0_cpy, nullptr, nullptr, nullptr, hx, hy, hz,
                              1.0, sz, bflag);
        SOLVER_DERIVS->filter(A1, A1_cpy, nullptr, nullptr, nullptr, hx, hy, hz,
                              1.0, sz, bflag);
        SOLVER_DERIVS->filter(A2, A2_cpy, nullptr, nullptr, nullptr, hx, hy, hz,
                              1.0, sz, bflag);
        SOLVER_DERIVS->filter(Gamma, Gamma_cpy, nullptr, nullptr, nullptr, hx, hy, hz,
                              1.0, sz, bflag);
        SOLVER_DERIVS->filter(psi, psi_cpy, nullptr, nullptr, nullptr, hx, hy, hz,
                              1.0, sz, bflag);
        SOLVER_DERIVS->filter(phi, phi_cpy, nullptr, nullptr, nullptr, hx, hy, hz,
                              1.0, sz, bflag);
    }

    // calculate the derivatives, on the copies if necessary
    SOLVER_DERIVS->grad_x(grad_0_E0, E0_cpy, hx, sz, bflag);
    SOLVER_DERIVS->grad_y(grad_1_E0, E0_cpy, hy, sz, bflag);  // needed
    SOLVER_DERIVS->grad_z(grad_2_E0, E0_cpy, hz, sz, bflag);  // needed

    SOLVER_DERIVS->grad_x(grad_0_E1, E1_cpy, hx, sz, bflag);  // needed
    SOLVER_DERIVS->grad_y(grad_1_E1, E1_cpy, hy, sz, bflag);
    SOLVER_DERIVS->grad_z(grad_2_E1, E1_cpy, hz, sz, bflag);  // needed

    SOLVER_DERIVS->grad_x(grad_0_E2, E2_cpy, hx, sz, bflag);  // needed
    SOLVER_DERIVS->grad_y(grad_1_E2, E2_cpy, hy, sz, bflag);  // needed
    SOLVER_DERIVS->grad_z(grad_2_E2, E2_cpy, hz, sz, bflag);

    SOLVER_DERIVS->grad_x(grad_0_A0, A0_cpy, hx, sz, bflag);
    SOLVER_DERIVS->grad_y(grad_1_A0, A0_cpy, hy, sz, bflag);  // needed
    SOLVER_DERIVS->grad_z(grad_2_A0, A0_cpy, hz, sz, bflag);  // needed

    SOLVER_DERIVS->grad_x(grad_0_A1, A1_cpy, hx, sz, bflag);  // needed
    SOLVER_DERIVS->grad_y(grad_1_A1, A1_cpy, hy, sz, bflag);
    SOLVER_DERIVS->grad_z(grad_2_A1, A1_cpy, hz, sz, bflag);  // needed

    SOLVER_DERIVS->grad_x(grad_0_A2, A2_cpy, hx, sz, bflag);  // needed
    SOLVER_DERIVS->grad_y(grad_1_A2, A2_cpy, hy, sz, bflag);  // needed
    SOLVER_DERIVS->grad_z(grad_2_A2, A2_cpy, hz, sz, bflag);

    SOLVER_DERIVS->grad_x(grad_0_Gamma, Gamma_cpy, hx, sz, bflag);  // needed
    SOLVER_DERIVS->grad_y(grad_1_Gamma, Gamma_cpy, hy, sz, bflag);  // needed
    SOLVER_DERIVS->grad_z(grad_2_Gamma, Gamma_cpy, hz, sz, bflag);

    SOLVER_DERIVS->grad_x(grad_0_psi, psi_cpy, hx, sz, bflag);
    SOLVER_DERIVS->grad_y(grad_1_psi, psi_cpy, hy, sz, bflag);
    SOLVER_DERIVS->grad_z(grad_2_psi, psi_cpy, hz, sz, bflag);

    SOLVER_DERIVS->grad_x(grad_0_phi, phi_cpy, hx, sz, bflag);
    SOLVER_DERIVS->grad_y(grad_1_phi, phi_cpy, hy, sz, bflag);
    SOLVER_DERIVS->grad_z(grad_2_phi, phi_cpy, hz, sz, bflag);

    SOLVER_DERIVS->grad_xx(grad2_0_0_psi, psi_cpy, hx, sz, bflag); 
    SOLVER_DERIVS->grad_yy(grad2_1_1_psi, psi_cpy, hy, sz, bflag); 
    SOLVER_DERIVS->grad_zz(grad2_2_2_psi, psi_cpy, hz, sz, bflag); 

        // 2nd derivs for A0. 
    SOLVER_DERIVS->grad_xx(grad2_0_0_A0, A0_cpy, hx, sz, bflag);
    SOLVER_DERIVS->grad_yy(grad2_1_1_A0, A0_cpy, hy, sz, bflag);
    SOLVER_DERIVS->grad_zz(grad2_2_2_A0, A0_cpy, hz, sz, bflag);

    // 2nd derivs for A1
    SOLVER_DERIVS->grad_xx(grad2_0_0_A1, A1_cpy, hx, sz, bflag);
    SOLVER_DERIVS->grad_yy(grad2_1_1_A1, A1_cpy, hy, sz, bflag);
    SOLVER_DERIVS->grad_zz(grad2_2_2_A1, A1_cpy, hz, sz, bflag);

    // 2nd derivs for A2
    SOLVER_DERIVS->grad_xx(grad2_0_0_A2, A2_cpy, hx, sz, bflag); 
    SOLVER_DERIVS->grad_yy(grad2_1_1_A2, A2_cpy, hy, sz, bflag);
    SOLVER_DERIVS->grad_zz(grad2_2_2_A2, A2_cpy, hz, sz, bflag);

    dsolve::timer::t_deriv.stop();

    // TODO: is this even necessary?
    double *rho_e = __mem_pool->allocate(n);

    double *J0 = __mem_pool->allocate(n);
    double *J1 = __mem_pool->allocate(n);
    double *J2 = __mem_pool->allocate(n);

    for (unsigned int m = 0; m < n; m++) {
        rho_e[m] = 0.0;

        J0[m] = 0.0;
        J1[m] = 0.0;
        J2[m] = 0.0;
    }

    // loop dep. removed allowing compiler to optmize for vectorization.
    dsolve::timer::t_rhs.start();
    for (unsigned int k = PW; k < nz - PW; k++) {
        for (unsigned int j = PW; j < ny - PW; j++) {
#ifdef SOLVER_ENABLE_AVX
#ifdef __INTEL_COMPILER
#pragma vector vectorlength(__RHS_AVX_SIMD_LEN__) vecremainder
#pragma ivdep
#endif
#endif
            for (unsigned int i = PW; i < nx - PW; i++) {
                const double x = pmin[0] + i * hx;
                const double y = pmin[1] + j * hy;
                const double z = pmin[2] + k * hz;

                const unsigned int pp = i + nx * (j + ny * k);
                const double r_coord = sqrt(x * x + y * y + z * z);

                double kappa_1 = 0.1 ; 
                double kappa_2 = 0.1 ;   



                // NOTE: (for now) we are not sending in the filtered E's and
                // B's if they're used. (They're not! But it's good to note!)
#include "../gencode/solver_rhs_eqns.cpp.inc"
            }
        }
    }
    dsolve::timer::t_rhs.stop();

    if (bflag != 0) {
        dsolve::timer::t_bdyc.start();

        asymptotic_and_falloff_bcs(E_rhs0, E0, grad_0_E0, grad_1_E0, grad_2_E0,
                                   pmin, pmax, 2.0, 0.0, sz, bflag);
        asymptotic_and_falloff_bcs(E_rhs1, E1, grad_0_E1, grad_1_E1, grad_2_E1,
                                   pmin, pmax, 2.0, 0.0, sz, bflag);
        asymptotic_and_falloff_bcs(E_rhs2, E2, grad_0_E2, grad_1_E2, grad_2_E2,
                                   pmin, pmax, 2.0, 0.0, sz, bflag);

        asymptotic_and_falloff_bcs(A_rhs0, A0, grad_0_A0, grad_1_A0, grad_2_A0,
                                   pmin, pmax, 1.0, 0.0, sz, bflag);
        asymptotic_and_falloff_bcs(A_rhs1, A1, grad_0_A1, grad_1_A1, grad_2_A1,
                                   pmin, pmax, 1.0, 0.0, sz, bflag);
        asymptotic_and_falloff_bcs(A_rhs2, A2, grad_0_A2, grad_1_A2, grad_2_A2,
                                   pmin, pmax, 1.0, 0.0, sz, bflag);
        asymptotic_and_falloff_bcs(Gamma_rhs, Gamma, grad_0_Gamma, grad_1_Gamma, grad_2_Gamma,
                                   pmin, pmax, 2.0, 0.0, sz, bflag);
        asymptotic_and_falloff_bcs(psi_rhs, psi, grad_0_psi, grad_1_psi, grad_2_psi,
                                   pmin, pmax, 1.0, 0.0, sz, bflag);
        asymptotic_and_falloff_bcs(phi_rhs, phi, grad_0_phi, grad_1_phi, grad_2_phi,
                                   pmin, pmax, 1.0, 0.0, sz, bflag);


        //[[[end]]]

        dsolve::timer::t_bdyc.stop();
    }

    if (!SOLVER_DERIVS->do_filter_before()) {
        dsolve::timer::t_deriv.start();
        // TODO: include more types of build options

        // TODO: support for CFD calculation of explicit KO derivs
#include "../gencode/solver_rhs_ko_deriv_calc.cpp.inc"
        dsolve::timer::t_deriv.stop();

        dsolve::timer::t_rhs.start();

        const double sigma = KO_DISS_SIGMA;

        SOLVER_DERIVS->filter(E0, E_rhs0, grad_0_E0, grad_1_E0, grad_2_E0, hx,
                              hy, hz, sigma, sz, bflag);
        SOLVER_DERIVS->filter(E1, E_rhs1, grad_0_E1, grad_1_E1, grad_2_E1, hx,
                              hy, hz, sigma, sz, bflag);
        SOLVER_DERIVS->filter(E2, E_rhs2, grad_0_E2, grad_1_E2, grad_2_E2, hx,
                              hy, hz, sigma, sz, bflag);

        SOLVER_DERIVS->filter(A0, A_rhs0, grad_0_A0, grad_1_A0, grad_2_A0, hx,
                              hy, hz, sigma, sz, bflag);
        SOLVER_DERIVS->filter(A1, A_rhs1, grad_0_A1, grad_1_A1, grad_2_A1, hx,
                              hy, hz, sigma, sz, bflag);
        SOLVER_DERIVS->filter(A2, A_rhs2, grad_0_A2, grad_1_A2, grad_2_A2, hx,
                              hy, hz, sigma, sz, bflag);
        SOLVER_DERIVS->filter(Gamma, Gamma_rhs, grad_0_Gamma, grad_1_Gamma, grad_2_Gamma, hx,
                              hy, hz, sigma, sz, bflag);
        SOLVER_DERIVS->filter(psi, psi_rhs, grad_0_psi, grad_1_psi, grad_2_psi, hx,
                              hy, hz, sigma, sz, bflag);
        SOLVER_DERIVS->filter(phi, phi_rhs, grad_0_phi, grad_1_phi, grad_2_phi, hx,
                              hy, hz, sigma, sz, bflag);


        dsolve::timer::t_rhs.stop();
    }

    dsolve::timer::t_deriv.start();
    __mem_pool->free(J0);
    __mem_pool->free(J1);
    __mem_pool->free(J2);

    __mem_pool->free(rho_e);
    dsolve::timer::t_deriv.stop();
}

/*----------------------------------------------------------------------;
 *
 *
 *
 *----------------------------------------------------------------------*/
void asymptotic_and_falloff_bcs(double *f_rhs, const double *f,
                                const double *dxf, const double *dyf,
                                const double *dzf, const double *pmin,
                                const double *pmax, const double f_falloff,
                                const double f_asymptotic,
                                const unsigned int *sz,
                                const unsigned int &bflag) {
    const unsigned int nx = sz[0];
    const unsigned int ny = sz[1];
    const unsigned int nz = sz[2];

    const double hx = (pmax[0] - pmin[0]) / (nx - 1);
    const double hy = (pmax[1] - pmin[1]) / (ny - 1);
    const double hz = (pmax[2] - pmin[2]) / (nz - 1);

    const unsigned int PW = dsolve::SOLVER_PADDING_WIDTH;

    unsigned int ib = PW;
    unsigned int jb = PW;
    unsigned int kb = PW;
    unsigned int ie = sz[0] - PW;
    unsigned int je = sz[1] - PW;
    unsigned int ke = sz[2] - PW;

    double x, y, z;
    unsigned int pp;
    double inv_r;

    if (bflag & (1u << OCT_DIR_LEFT)) {
        double x = pmin[0] + ib * hx;
        for (unsigned int k = kb; k < ke; k++) {
            z = pmin[2] + k * hz;
            for (unsigned int j = jb; j < je; j++) {
                y = pmin[1] + j * hy;
                pp = IDX(ib, j, k);
                inv_r = 1.0 / sqrt(x * x + y * y + z * z);

                double f_temp = f_rhs[pp];

                f_rhs[pp] = -inv_r * (x * dxf[pp] + y * dyf[pp] + z * dzf[pp] +
                                      f_falloff * (f[pp] - f_asymptotic));

                // std::cout << "Testing boundary on OCT_DIR_LEFT: originally "
                // << f_temp << " is now " << f_rhs[pp] << " from " << x << ", "
                // << y << ", " << z << " with: falloff=" << f_falloff << " and
                // asymptotic=" << f_asymptotic << std::endl;
            }
        }
    }

    if (bflag & (1u << OCT_DIR_RIGHT)) {
        x = pmin[0] + (ie - 1) * hx;
        for (unsigned int k = kb; k < ke; k++) {
            z = pmin[2] + k * hz;
            for (unsigned int j = jb; j < je; j++) {
                y = pmin[1] + j * hy;
                pp = IDX((ie - 1), j, k);
                inv_r = 1.0 / sqrt(x * x + y * y + z * z);

                f_rhs[pp] = -inv_r * (x * dxf[pp] + y * dyf[pp] + z * dzf[pp] +
                                      f_falloff * (f[pp] - f_asymptotic));
            }
        }
    }

    if (bflag & (1u << OCT_DIR_DOWN)) {
        y = pmin[1] + jb * hy;
        for (unsigned int k = kb; k < ke; k++) {
            z = pmin[2] + k * hz;
            for (unsigned int i = ib; i < ie; i++) {
                x = pmin[0] + i * hx;
                inv_r = 1.0 / sqrt(x * x + y * y + z * z);
                pp = IDX(i, jb, k);

                f_rhs[pp] = -inv_r * (x * dxf[pp] + y * dyf[pp] + z * dzf[pp] +
                                      f_falloff * (f[pp] - f_asymptotic));
            }
        }
    }

    if (bflag & (1u << OCT_DIR_UP)) {
        y = pmin[1] + (je - 1) * hy;
        for (unsigned int k = kb; k < ke; k++) {
            z = pmin[2] + k * hz;
            for (unsigned int i = ib; i < ie; i++) {
                x = pmin[0] + i * hx;
                inv_r = 1.0 / sqrt(x * x + y * y + z * z);
                pp = IDX(i, (je - 1), k);

                f_rhs[pp] = -inv_r * (x * dxf[pp] + y * dyf[pp] + z * dzf[pp] +
                                      f_falloff * (f[pp] - f_asymptotic));
            }
        }
    }

    if (bflag & (1u << OCT_DIR_BACK)) {
        z = pmin[2] + kb * hz;
        for (unsigned int j = jb; j < je; j++) {
            y = pmin[1] + j * hy;
            for (unsigned int i = ib; i < ie; i++) {
                x = pmin[0] + i * hx;
                inv_r = 1.0 / sqrt(x * x + y * y + z * z);
                pp = IDX(i, j, kb);

                f_rhs[pp] = -inv_r * (x * dxf[pp] + y * dyf[pp] + z * dzf[pp] +
                                      f_falloff * (f[pp] - f_asymptotic));
            }
        }
    }

    if (bflag & (1u << OCT_DIR_FRONT)) {
        z = pmin[2] + (ke - 1) * hz;
        for (unsigned int j = jb; j < je; j++) {
            y = pmin[1] + j * hy;
            for (unsigned int i = ib; i < ie; i++) {
                x = pmin[0] + i * hx;
                inv_r = 1.0 / sqrt(x * x + y * y + z * z);
                pp = IDX(i, j, (ke - 1));

                f_rhs[pp] = -inv_r * (x * dxf[pp] + y * dyf[pp] + z * dzf[pp] +
                                      f_falloff * (f[pp] - f_asymptotic));
            }
        }
    }
}

// TODO: boundary conditions for reflective box

/*----------------------------------------------------------------------;
 *
 *
 *
 *----------------------------------------------------------------------*/
void max_spacetime_speeds(double *const lambda1max, double *const lambda2max,
                          double *const lambda3max, const double *const alpha,
                          const double *const beta1, const double *const beta2,
                          const double *const beta3, const double *const gtd11,
                          const double *const gtd12, const double *const gtd13,
                          const double *const gtd22, const double *const gtd23,
                          const double *const gtd33, const double *const chi,
                          const unsigned int *sz) {
    const unsigned int nx = sz[0];
    const unsigned int ny = sz[1];
    const unsigned int nz = sz[2];

    const unsigned int PW = dsolve::SOLVER_PADDING_WIDTH;
    unsigned int ib = PW;
    unsigned int jb = PW;
    unsigned int kb = PW;
    unsigned int ie = sz[0] - PW;
    unsigned int je = sz[1] - PW;
    unsigned int ke = sz[2] - PW;

    for (unsigned int k = kb; k < ke; k++) {
        for (unsigned int j = jb; j < je; j++) {
            for (unsigned int i = ib; i < ie; i++) {
                unsigned int pp = IDX(i, j, k);
                /* note: gtu is the inverse tilde metric. It should have detgtd
                 * = 1. So, for the purposes of
                 * calculating wavespeeds, I simple set detgtd = 1. */
                double gtu11 = gtd22[pp] * gtd33[pp] - gtd23[pp] * gtd23[pp];
                double gtu22 = gtd11[pp] * gtd33[pp] - gtd13[pp] * gtd13[pp];
                double gtu33 = gtd11[pp] * gtd22[pp] - gtd12[pp] * gtd12[pp];
                if (gtu11 < 0.0 || gtu22 < 0.0 || gtu33 < 0.0) {
                    std::cout << "Problem computing spacetime characteristics"
                              << std::endl;
                    std::cout << "gtu11 = " << gtu11 << ", gtu22 = " << gtu22
                              << ", gtu33 = " << gtu33 << std::endl;
                    gtu11 = 1.0;
                    gtu22 = 1.0;
                    gtu33 = 1.0;
                }
                double t1 = alpha[pp] * sqrt(gtu11 * chi[pp]);
                double t2 = alpha[pp] * sqrt(gtu22 * chi[pp]);
                double t3 = alpha[pp] * sqrt(gtu33 * chi[pp]);
                lambda1max[pp] =
                    std::max(abs(-beta1[pp] + t1), abs(-beta1[pp] - t1));
                lambda2max[pp] =
                    std::max(abs(-beta2[pp] + t2), abs(-beta2[pp] - t2));
                lambda3max[pp] =
                    std::max(abs(-beta3[pp] + t3), abs(-beta3[pp] - t3));
            }
        }
    }
}

/*----------------------------------------------------------------------;
 *
 * Forces RHS boundaries to be zero
 *
 *----------------------------------------------------------------------*/
void freeze_bcs(double *f_rhs, const unsigned int *sz,
                const unsigned int &bflag) {
    const unsigned int nx = sz[0];
    const unsigned int ny = sz[1];
    const unsigned int nz = sz[2];

    const unsigned int PW = dsolve::SOLVER_PADDING_WIDTH;
    unsigned int ib = PW;
    unsigned int jb = PW;
    unsigned int kb = PW;
    unsigned int ie = sz[0] - PW;
    unsigned int je = sz[1] - PW;
    unsigned int ke = sz[2] - PW;

    unsigned int pp;

    if (bflag & (1u << OCT_DIR_LEFT)) {
        for (unsigned int k = kb; k < ke; k++) {
            for (unsigned int j = jb; j < je; j++) {
                pp = IDX(ib, j, k);
                f_rhs[pp] = 0.0;
            }
        }
    }

    if (bflag & (1u << OCT_DIR_RIGHT)) {
        for (unsigned int k = kb; k < ke; k++) {
            for (unsigned int j = jb; j < je; j++) {
                pp = IDX((ie - 1), j, k);
                f_rhs[pp] = 0.0;
            }
        }
    }

    if (bflag & (1u << OCT_DIR_DOWN)) {
        for (unsigned int k = kb; k < ke; k++) {
            for (unsigned int i = ib; i < ie; i++) {
                pp = IDX(i, jb, k);
                f_rhs[pp] = 0.0;
            }
        }
    }

    if (bflag & (1u << OCT_DIR_UP)) {
        for (unsigned int k = kb; k < ke; k++) {
            for (unsigned int i = ib; i < ie; i++) {
                pp = IDX(i, (je - 1), k);
                f_rhs[pp] = 0.0;
            }
        }
    }

    if (bflag & (1u << OCT_DIR_BACK)) {
        for (unsigned int j = jb; j < je; j++) {
            for (unsigned int i = ib; i < ie; i++) {
                pp = IDX(i, j, kb);
                f_rhs[pp] = 0.0;
            }
        }
    }

    if (bflag & (1u << OCT_DIR_FRONT)) {
        for (unsigned int j = jb; j < je; j++) {
            for (unsigned int i = ib; i < ie; i++) {
                pp = IDX(i, j, (ke - 1));
                f_rhs[pp] = 0.0;
            }
        }
    }
}
