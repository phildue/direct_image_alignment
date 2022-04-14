#include <memory>

#include "utils/utils.h"
#include "core/core.h"
namespace pd::vslam::solver{

    template<int nParameters>
    GaussNewton<nParameters>::GaussNewton(
            double minStepSize,
            int maxIterations
            )
    :_minStepSize(minStepSize)
    ,_minGradient(minStepSize)
    ,_minReduction(minStepSize)
    ,_maxIterations(maxIterations)

    {
        vision::Log::get("solver",SOLVER_CFG_DIR"/log/solver.conf");
        _chi2 = Eigen::VectorXd::Zero(_maxIterations);
        _stepSize = Eigen::VectorXd::Zero(_maxIterations);
        _x = Eigen::MatrixXd::Zero(_maxIterations,nParameters);
        _i = 0;
    }
  

    template<int nParameters>
    void GaussNewton<nParameters>::solve(std::shared_ptr< Problem<nParameters> > problem) {
        
        SOLVER( INFO ) << "Solving Problem for " << nParameters << " parameters.";
        TIMED_FUNC(timerF);
        _chi2.setZero();
        _stepSize.setZero();
        _x.setZero();
        for(_i = 0; _i < _maxIterations; _i++ )
        {
            TIMED_SCOPE(timerI,"solve ( " + std::to_string(_i) + " )");
            // We want to solve dx = (JWJ)^(-1)*JWr
            // This can be solved with cholesky decomposition (Ax = b)
            // Where A = (JWJ + lambda * I), x = dx, b = JWr
            auto ne = problem->computeNormalEquations();
            SOLVER(DEBUG) << _i << " > A=\n" << ne->A << "\nb=\n" << ne->b.transpose() << "\nnConstraints="<< ne->nConstraints << " chi2=" <<ne->chi2;

            _chi2(_i) = ne->chi2 / ne->nConstraints;

            const double dChi2 = _i > 0 ? _chi2(_i)-_chi2(_i-1) : 0;
            if (_i > 0 && dChi2 > 0)
            {
                SOLVER( INFO ) << _i << " > " << "CONVERGED. No improvement";
                problem->setX(_x.row(_i-1));
                break;
            }
            //TODO normalization necessary?
            const auto A = ne->A;
            const auto b = ne->b;
            const auto dx = A.ldlt().solve(b);
            _H  = ne->A;
            problem->updateX(dx);
            _x.row(_i) = problem->x();
            _stepSize(_i) = dx.norm();

            SOLVER( INFO ) << "Iteration: " << _i << " chi2: " << _chi2(_i) << " dChi2: " << dChi2 << " stepSize: " << _stepSize(_i) 
            << " Points: " << ne->nConstraints << "\nx: " << problem->x().transpose() << "\ndx: " << dx.transpose();
            if ( _i > 0 && (_stepSize(_i) < _minStepSize || std::abs(ne->b.maxCoeff()) < _minGradient || std::abs(dChi2) < _minReduction) )
            { 
                SOLVER( INFO ) << _i << " > " << _stepSize(_i) << "/" << _minStepSize << " CONVERGED. ";
                break;
            }

            if (!std::isfinite(_stepSize(_i)))
            {
                throw pd::Exception(std::to_string(_i) + "> NaN during optimization.");
            }

        }
        LOG_PLT("SolverGN") << std::make_shared<pd::vision::vis::PlotGaussNewton>(_i,_chi2,_stepSize);

    }

}