#include <memory>

#include "utils/utils.h"
#include "core/core.h"
namespace pd{namespace vision{

    template<typename Problem>
    GaussNewton<Problem>::GaussNewton(
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
        Log::get("solver",SOLVER_CFG_DIR"/log/solver.conf");
        _chi2 = Eigen::VectorXd::Zero(_maxIterations);
        _stepSize = Eigen::VectorXd::Zero(_maxIterations);
        _x = Eigen::MatrixXd::Zero(_maxIterations,Problem::nParameters);
        _i = 0;
    }
  

    template<typename Problem>
    void GaussNewton<Problem>::solve(std::shared_ptr< Problem> problem) {
        
        SOLVER( INFO ) << "Solving Problem for " << Problem::nParameters << " parameters.";
        TIMED_FUNC(timerF);
        _chi2.setZero();
        _stepSize.setZero();
        _x.setZero();
        Eigen::Vector<double, Problem::nParameters> dx;
        for(_i = 0; _i < _maxIterations; _i++ )
        {
            TIMED_SCOPE(timerI,"solve ( " + std::to_string(_i) + " )");


            // We want to solve dx = (JWJ)^(-1)*JWr
            // This can be solved with cholesky decomposition (Ax = b)
            // Where A = (JWJ + lambda * I), x = dx, b = JWr

            Eigen::VectorXd r,w;
            problem->computeResidual(r,w);
            const auto W = w.asDiagonal();
         
            LOG_PLT("ErrorDistribution") << std::make_shared<vis::Histogram>(r,"Residuals");

            _chi2(_i) = (r.transpose() * W * r);
            _chi2(_i) /= r.rows();
            const double dChi2 = _i > 0 ? _chi2(_i)-_chi2(_i-1) : 0;
            if (_i > 0 && dChi2 > 0)
            {
                SOLVER( INFO ) << _i << " > " << "CONVERGED. No improvement";
                problem->updateX(-dx);
                break;
            }
            Mmxn J = Eigen::MatrixXd::Zero(r.rows(), Problem::nParameters);
            problem->computeJacobian(J);
            // For GN / LM we drop the second part of the Hessian
            _H  = (J.transpose() * W * J);
            problem->extendLeft(_H);//User can provide additional conditions TODO find better name?

        
            SOLVER(DEBUG) << _i << " > H.:\n" << _H;
        
            Eigen::VectorXd gradient = J.transpose() * W * r;
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
        LOG_PLT("SolverGN") << std::make_shared<vis::PlotGaussNewton>(_i,_chi2,_stepSize);

    }



}}