#include <memory>

#include "utils/utils.h"
#include "core/core.h"
namespace pd::vslam::solver{

    template<int nParameters>
    GaussNewton<nParameters>::GaussNewton(
            double alpha,
            double minStepSize,
            int maxIterations
            )
    :_alpha(alpha)
    ,_minStepSize(minStepSize)
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
        Eigen::Vector<double, nParameters> dx;
        Mmxn J = vision::MatXd::Zero(problem->nConstraints(),nParameters);
        problem->computeJacobian(J,0U);
        Eigen::VectorXd r = vision::VecXd::Zero(problem->nConstraints()), w = vision::VecXd::Zero(problem->nConstraints());
        for(_i = 0; _i < _maxIterations; _i++ )
        {
            TIMED_SCOPE(timerI,"solve ( " + std::to_string(_i) + " )");


            // We want to solve dx = (JWJ)^(-1)*JWr
            // This can be solved with cholesky decomposition (Ax = b)
            // Where A = (JWJ + lambda * I), x = dx, b = JWr

            problem->computeResidual(r,w,0U);
            const auto W = w.asDiagonal();
         
            LOG_PLT("ErrorDistribution") << std::make_shared<vision::vis::Histogram>(r,"Residuals");

            _chi2(_i) = (r.transpose() * W * r);
            _chi2(_i) /= r.rows();
            const double dChi2 = _i > 0 ? _chi2(_i)-_chi2(_i-1) : 0;
            if (_i > 0 && dChi2 > 0)
            {
                SOLVER( INFO ) << _i << " > " << "CONVERGED. No improvement";
                problem->updateX(-dx);
                break;
            }
            if(problem->newJacobian())
            {
                problem->computeJacobian(J,0U);
            }
            // For GN / LM we drop the second part of the Hessian
            _H  = (J.transpose() * W * J);
            problem->extendLeft(_H);//User can provide additional conditions TODO find better name?

        
            SOLVER(DEBUG) << _i << " > H.:\n" << _H;
        
            Eigen::Vector<double,nParameters> gradient = J.transpose() * W * r;
            problem->extendRight(gradient);//User can provide additional conditions TODO find better name?

            SOLVER(DEBUG) << _i << " > Grad.:\n" << gradient.transpose() ;
            _H /= r.rows();
            gradient /= r.rows();

            dx = _alpha * _H.ldlt().solve( gradient );

            SOLVER(DEBUG) << _i <<" > x:\n" << problem->x().transpose() ;
            SOLVER(DEBUG) << _i <<" > dx:\n" << dx.transpose() ;
            problem->updateX(dx);
            _x.row(_i) = problem->x();
            _stepSize(_i) = dx.norm();

            SOLVER( INFO ) << "Iteration: " << _i << " chi2: " << _chi2(_i) << " dChi2: " << dChi2 << " stepSize: " << _stepSize(_i) << " Points: " << r.rows() << " x: " << problem->x().transpose() << " dx: " << dx.transpose();
            if ( _i > 0 && (_stepSize(_i) < _minStepSize || std::abs(gradient.maxCoeff()) < _minGradient || std::abs(dChi2) < _minReduction) )
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