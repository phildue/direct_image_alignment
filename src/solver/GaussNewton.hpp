#include <memory>

#include "utils/visuals.h"

#include "utils/Log.h"
#include "utils/Exceptions.h"
#include "core/algorithm.h"
namespace pd{namespace vision{
    #define LOG_PLOT_GN(name) Log::getPlotLog(name,Level::Debug)

    template<typename Problem>
    GaussNewton<Problem>::GaussNewton(
            double alpha,
            double minStepSize,
            int maxIterations
            )
    :_alpha(alpha)
    ,_minStepSize(minStepSize)
    ,_maxIterations(maxIterations)
    ,_minGradient(minStepSize)
    ,_minReduction(minStepSize)
    {
        Log::get("solver");
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
        for(_i = 0; _i < _maxIterations; _i++ )
        {
            TIMED_SCOPE(timerI,"solve ( " + std::to_string(_i) + " )");


            // We want to solve dx = (JWJ)^(-1)*JWr
            // This can be solved with cholesky decomposition (Ax = b)
            // Where A = (JWJ + lambda * I), x = dx, b = JWr

            Eigen::VectorXd r,w;
            problem->computeResidual(r,w);
            const auto W = w.asDiagonal();
         
            LOG_PLOT_GN("ErrorDistribution") << std::make_shared<vis::Histogram>(r,"Residuals");

            _chi2(_i) = r.transpose() * W * r;
            Mmxn J = Eigen::MatrixXd::Zero(r.rows(), Problem::nParameters);
            problem->computeJacobian(J);
            // For GN / LM we drop the second part of the Hessian
            const Eigen::MatrixXd H  = (J.transpose() * W * J);
        
            SOLVER(DEBUG) << _i << " > H.:\n" << H;
        
            const Eigen::VectorXd gradient = J.transpose() * W * r;

            SOLVER(DEBUG) << _i << " > Grad.:\n" << gradient.transpose() ;

            const Eigen::Vector<double, Eigen::Dynamic> dx = _alpha * H.ldlt().solve( gradient );

            SOLVER(DEBUG) << _i <<" > x:\n" << problem->x().transpose() ;
            SOLVER(DEBUG) << _i <<" > dx:\n" << dx.transpose() ;
            problem->updateX(dx);
            _x.row(_i) = problem->x();
            _stepSize(_i) = dx.norm();

            const double dChi2 = _i > 0 ? _chi2(_i)-_chi2(_i-1) : 0;
            SOLVER( INFO ) << "Iteration: " << _i << " chi2: " << _chi2(_i) << " dChi2: " << dChi2 << " stepSize: " << _stepSize(_i) << " Valid Points: " << r.rows();
            if ( _i > 0 && (_stepSize(_i) < _minStepSize || std::abs(gradient.maxCoeff()) < _minGradient || std::abs(dChi2) < _minReduction))
            { 
                SOLVER( INFO ) << _i << " > " << _stepSize(_i) << "/" << _minStepSize << " CONVERGED. ";
                break;
            }

            if (!std::isfinite(_stepSize(_i)))
            {
                throw pd::Exception(std::to_string(_i) + "> NaN during optimization.");
            }

        }
        LOG_PLOT_GN("SolverGN") << std::make_shared<vis::PlotGaussNewton>(_i,_chi2,_stepSize);

    }



}}