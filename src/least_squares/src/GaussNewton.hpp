#include <memory>

#include "utils/utils.h"
#include "core/core.h"
namespace pd::vslam::least_squares{

    template<int nParameters>
    GaussNewton<nParameters>::GaussNewton(
            double minStepSize,
            size_t maxIterations
            )
    :_minStepSize(minStepSize)
    ,_minGradient(minStepSize)
    ,_minReduction(minStepSize)
    ,_maxIterations(maxIterations)

    {
        Log::get("solver",LEAST_SQUARES_CFG_DIR"/log/solver.conf");
        
    }
  

    template<int nParameters>
    typename Solver<nParameters>::Results::ConstUnPtr GaussNewton<nParameters>::solve(std::shared_ptr< Problem<nParameters> > problem) {
        
        SOLVER( INFO ) << "Solving Problem for " << nParameters << " parameters.";
        TIMED_FUNC(timerF);

        auto r = std::make_unique<typename Solver<nParameters>::Results>();

        r->chi2 = Eigen::VectorXd::Zero(_maxIterations);
        r->stepSize = Eigen::VectorXd::Zero(_maxIterations);
        r->x = Eigen::MatrixXd::Zero(_maxIterations,nParameters);
        r->cov.reserve(_maxIterations);
        size_t i = 0;
        for(; i < _maxIterations; i++ )
        {
            TIMED_SCOPE(timerI,"solve ( " + std::to_string(i) + " )");
            // We want to solve dx = (JWJ)^(-1)*JWr
            // This can be solved with cholesky decomposition (Ax = b)
            // Where A = (JWJ + lambda * I), x = dx, b = JWr
            auto ne = problem->computeNormalEquations();

            const double det = ne->A.determinant();
            if(!std::isfinite(det) || det < 1e-6){
                SOLVER( WARNING ) << i << " > " << "STOP. Bad Hessian.";
                break;
            }
            SOLVER(DEBUG) << i << " > A=\n" << ne->A << "\nb=\n" << ne->b.transpose() << "\nnConstraints="<< ne->nConstraints << " chi2=" <<ne->chi2;

            r->chi2(i) = ne->chi2 / ne->nConstraints;

            const double dChi2 = i > 0 ? r->chi2(i)- r->chi2(i-1) : 0;
            if (i > 0 && dChi2 > 0 )
            {
                SOLVER( INFO ) << i << " > " << "CONVERGED. No improvement";
                problem->setX(r->x.row(i-1));
                break;
            }
            const auto dx = ne->A.ldlt().solve(ne->b);
            problem->updateX(dx);
            r->x.row(i) = problem->x();
            r->stepSize(i) = dx.norm();
            r->cov.push_back(ne->A.inverse());

            SOLVER( INFO ) << "Iteration: " << i << " chi2: " << r->chi2(i) << " dChi2: " << dChi2 << " stepSize: " << r->stepSize(i) 
            << " Points: " << ne->nConstraints << "\nx: " << problem->x().transpose() << "\ndx: " << dx.transpose();
            if ( i > 0 && (r->stepSize(i) < _minStepSize || std::abs(ne->b.maxCoeff()) < _minGradient || std::abs(dChi2) < _minReduction) )
            { 
                SOLVER( INFO ) << i << " > " << r->stepSize(i) << "/" << _minStepSize << " CONVERGED. ";
                break;
            }

            if (!std::isfinite(r->stepSize(i)))
            {
                throw pd::Exception(std::to_string(i) + "> NaN during optimization.");
            }

        }
        LOG_PLT("SolverGN") << std::make_shared<vis::PlotGaussNewton>(i,r->chi2,r->stepSize);
        return r;
    }

  

}