#include "utils/Log.h"
#include "utils/Exceptions.h"
#include "core/algorithm.h"

namespace pd{namespace vision{

    template<typename Problem, typename Loss>
    GaussNewton<Problem,Loss>::GaussNewton(
            int nObservations,
            double alpha,
            double minStepSize,
            int maxIterations
            )
    :_maxIterations(maxIterations)
    ,_nObservations(nObservations)
    ,_minStepSize(minStepSize)
    ,_alpha(alpha)
    {
        Log::get("solver");
    }
    template<typename Problem, typename Loss>
    void GaussNewton<Problem, Loss>::solve(std::shared_ptr< Problem> problem) const
    {
        Eigen::VectorXd chiSquared(_maxIterations);
        chiSquared.setZero();
        Eigen::VectorXd stepSize(_maxIterations);
        stepSize.setZero();
        Eigen::Matrix<double, Eigen::Dynamic, Problem::nParameters> x(_maxIterations,Problem::nParameters);
        x.setZero();

        solve(problem,chiSquared,stepSize,x);
    }

   

    template<typename Problem, typename Loss>
    void GaussNewton<Problem, Loss>::solve(std::shared_ptr< Problem> problem, Eigen::VectorXd &chi2, Eigen::VectorXd& stepSize, GaussNewton::Mmxn & x) const {
        
        SOLVER( INFO ) << "Solving Problem for " << Problem::nParameters << " parameters. With " << _nObservations << " observations.";
        TIMED_FUNC(timerF);
        
        int iLast = 0;
        for(int i = 0; i < _maxIterations; i++)
        {
            TIMED_SCOPE(timerI,"solve ( " + std::to_string(i) + " )");

            Eigen::VectorXd W = Eigen::VectorXd::Zero(_nObservations);
            Mmxn J = Eigen::MatrixXd::Zero(_nObservations, Problem::nParameters);
            Eigen::VectorXd residuals = Eigen::VectorXd::Zero(_nObservations);

            // We want to solve dx = (JWJ)^(-1)*JWr
            // This can be solved with cholesky decomposition (Ax = b)
            // Where A = (JWJ + lambda * I), x = dx, b = JWr

            problem->computeResidual(residuals);
            Loss::computeWeights(residuals,W);
            chi2(i) = (residuals.transpose() * W.asDiagonal() * residuals);

            problem->computeJacobian(J);
            // For GN / LM we drop the second part of the Hessian
            const Eigen::MatrixXd H  = (J.transpose() * W.asDiagonal() * J);
        
            SOLVER(DEBUG) << i << " > H.:\n" << H;
        
            const Eigen::VectorXd gradient = J.transpose() * W.asDiagonal() * residuals;

            SOLVER(DEBUG) << i << " > Grad.:\n" << gradient.transpose() ;

            const Eigen::Vector<double, Eigen::Dynamic> dx = _alpha * H.ldlt().solve( gradient );

            SOLVER(DEBUG) << i <<" > x:\n" << problem->x().transpose() ;
            SOLVER(DEBUG) << i <<" > dx:\n" << dx.transpose() ;
            problem->updateX(dx);
            x.row(i) = problem->x();
            stepSize(i) = dx.norm();

            const double dChi2 = i > 0 ? chi2(i)-chi2(i-1) : 0;
            SOLVER( INFO ) << "Iteration: " << i << " chi2: " << chi2(i) << " dChi2: " << dChi2 << " stepSize: " << stepSize(i) << " Total Weight: " << W.sum();
            if ( stepSize(i) < _minStepSize )
            {
                SOLVER( INFO ) << i << " > " << stepSize(i) << "/" << _minStepSize << " CONVERGED. ";
                break;
            }

            if (!std::isfinite(stepSize(i)))
            {
                throw pd::Exception(std::to_string(i) + "> NaN during optimization.");
            }


        }
    }

       
    }}