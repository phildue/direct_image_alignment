#include "utils/Log.h"
#include "utils/Exceptions.h"
#include "core/algorithm.h"

namespace pd{namespace vision{

    template<int nParameters>
    GaussNewton<nParameters>::GaussNewton(std::function<bool(const Eigen::Matrix<double, nParameters, 1>&, Eigen::VectorXd&, Eigen::VectorXd& )> computeResidual,
            std::function<bool(const Eigen::Matrix<double, nParameters, 1>&, Mmxn&)> computeJacobian,
            std::function<bool(const Eigen::Matrix<double, nParameters, 1>&, Eigen::Matrix<double, nParameters, 1>&)> updateX,
            int nObservations,
            double alpha,
            double minStepSize,
            int maxIterations
            )
    :_computeResidual(computeResidual)
    ,_computeJacobian(computeJacobian)
    ,_updateX(updateX)
    ,_maxIterations(maxIterations)
    ,_nObservations(nObservations)
    ,_nParameters(nParameters)
    ,_minStepSize(minStepSize)
    ,_alpha(alpha)
    {
        Log::get("solver");
    }
    template<int nParameters>
    void GaussNewton<nParameters>::solve(Eigen::Matrix<double, nParameters, 1>& x) const
    {
        Eigen::VectorXd chiSquared(_maxIterations);
        chiSquared.setZero();
        Eigen::VectorXd stepSize(_maxIterations);
        stepSize.setZero();
        solve(x,chiSquared,stepSize);
    }

   

    template<int nParameters>
    void GaussNewton<nParameters>::solve(Eigen::Matrix<double, nParameters, 1> &x, Eigen::VectorXd &chi2, Eigen::VectorXd& stepSize) const {
        
        SOLVER( INFO ) << "Solving Problem for " << _nParameters << " parameters. With " << _nObservations << " observations.";
        
        int iLast = 0;
        for(int i = 0; i < _maxIterations; i++)
        {
            Eigen::VectorXd W = Eigen::VectorXd::Zero(_nObservations);
            Mmxn J = Eigen::MatrixXd::Zero(_nObservations, _nParameters);
            Eigen::VectorXd residuals = Eigen::VectorXd::Zero(_nObservations);

            // We want to solve dx = (JWJ)^(-1)*JWr
            // This can be solved with cholesky decomposition (Ax = b)
            // Where A = (JWJ + lambda * I), x = dx, b = JWr

            _computeResidual(x,residuals,W);
            computeWeights(residuals,W);
            chi2(i) = (residuals.transpose() * W.asDiagonal() * residuals);

            _computeJacobian(x,J);
            // For GN / LM we drop the second part of the Hessian
            const Eigen::MatrixXd H  = (J.transpose() * W.asDiagonal() * J);
        
            SOLVER(DEBUG) << i << " > H.:\n" << H;
        
            const Eigen::VectorXd gradient = J.transpose() * W.asDiagonal() * residuals;

            SOLVER(DEBUG) << i << " > Grad.:\n" << gradient.transpose() ;

            const Eigen::VectorXd dx = _alpha * H.ldlt().solve( gradient );

            SOLVER(DEBUG) << i <<" > x:\n" << x.transpose() ;
            SOLVER(DEBUG) << i <<" > dx:\n" << dx.transpose() ;
            _updateX(dx,x);
            stepSize(i) = dx.norm();

            if ( stepSize(i) < _minStepSize )
            {
                SOLVER( INFO ) << i << " > " << stepSize(i) << "/" << _minStepSize << " CONVERGED. ";
                break;
            }

            if (!std::isfinite(stepSize(i)))
            {
                throw pd::Exception(std::to_string(i) + "> NaN during optimization.");
            }

            const double dChi2 = i > 0 ? chi2(i)-chi2(i-1) : 0;
            SOLVER( INFO ) << "Iteration: " << i << " chi2: " << chi2(i) << " dChi2: " << dChi2 << " stepSize: " << stepSize(i) << " Total Weight: " << W.sum();
            


        }
    }

    template<int nParameters>
    void GaussNewton<nParameters>::computeWeights(const Eigen::VectorXd& residuals, Eigen::VectorXd& weights) const
    {
                    
    }

       
    }}