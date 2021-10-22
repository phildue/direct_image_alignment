#include "LevenbergMarquardt.h"
#include "utils/Log.h"
#include "utils/Exceptions.h"
#include "core/algorithm.h"

namespace pd{namespace vision{

    LevenbergMarquardt::LevenbergMarquardt(std::function<bool(const Eigen::VectorXd&, Eigen::VectorXd&,Eigen::VectorXd&)> computeResidual,
            std::function<bool(const Eigen::VectorXd&, Eigen::MatrixXd&)> computeJacobian,
            std::function<bool(const Eigen::VectorXd&, Eigen::VectorXd&)> updateX,
            int nObservations,
            int nParameters,
            int maxIterations,
            double minStepSize,
            double minGradient
            )
    :_computeResidual(computeResidual)
    ,_computeJacobian(computeJacobian)
    ,_updateX(updateX)
    ,_maxIterations(maxIterations)
    ,_nObservations(nObservations)
    ,_nParameters(nParameters)
    ,_minStepSize(minStepSize)
    ,_minGradient(minGradient)
    {
        Log::get("solver");
    }
    void LevenbergMarquardt::solve(Eigen::VectorXd& x)
    {
        Eigen::VectorXd chiSquared(_maxIterations);
        chiSquared.setZero();
        Eigen::VectorXd chi2Predicted(_maxIterations);
        chi2Predicted.setZero();
       
        Eigen::VectorXd lambda(_maxIterations);
        lambda.setZero();
        Eigen::VectorXd stepSize(_maxIterations);
        stepSize.setZero();
        solve(x,chiSquared,chi2Predicted,lambda,stepSize);
    }

    

        void LevenbergMarquardt::solve(Eigen::VectorXd &x, Eigen::VectorXd &chi2,Eigen::VectorXd &dchi2pred, Eigen::VectorXd &lambda, Eigen::VectorXd& stepSize) {
             SOLVER( INFO ) << "Solving Problem for " << _nParameters << " parameters. With " << _nObservations << " observations.";

            auto xprev = x;
     
            Eigen::VectorXd W = Eigen::VectorXd::Zero(_nObservations);
            Eigen::MatrixXd J = Eigen::MatrixXd::Zero(_nObservations, _nParameters);
            Eigen::VectorXd r = Eigen::VectorXd::Zero(_nObservations);
            
            // We want to solve dx = (JWJ)^(-1)*JWr
                // This can be solved with cholesky decomposition (Ax = b)
                // Where A = (JWJ + lambda * I), x = dx, b = JWr

            _computeResidual(x,r,W);
            computeWeights(r,W);
            chi2(0) = (r.transpose() * W.asDiagonal() * r);
            _computeJacobian(x,J);
            // For GN / LM we drop the second part of the Hessian
            Eigen::MatrixXd H  = (J.transpose() * W.asDiagonal() * J);
            double chi2Last;
            for(int i = 0; i < _maxIterations -1; i++)
            {
                if ( i == 0)
                {
                    lambda(i) = H.norm();
                }
                // Lagrange multiplier steers magnitude and direction of the step
                // For lambda ~ 0 the update will be Gauss-Newton
                // For large lambda the update will be gradient
                H.noalias() += lambda(i)*Eigen::MatrixXd::Identity(J.cols(),J.cols());
               
                SOLVER(DEBUG) << i << " > H.:\n" << H;
            
                const Eigen::VectorXd gradient = J.transpose() * W.asDiagonal() * r;

                SOLVER(DEBUG) << i << " > Grad.:\n" << gradient.transpose() ;

                const Eigen::VectorXd dx = H.ldlt().solve( gradient );

                SOLVER(DEBUG) << i <<" > x:\n" << x.transpose() ;
                SOLVER(DEBUG) << i <<" > dx:\n" << dx.transpose() ;
                _updateX(dx,x);
                stepSize(i) = dx.norm();

                if ( stepSize(i) < _minStepSize || std::abs(gradient.maxCoeff()) < _minGradient)
                {
                    SOLVER( INFO ) << i << " > Stepsize: " << stepSize(i) << "/" << _minStepSize << 
                    " Gradient: " << gradient.maxCoeff() << "/" << _minGradient << " CONVERGED. ";
                    break;
                }

                if (!std::isfinite(stepSize(i)))
                {
                    throw pd::Exception(std::to_string(i) + "> NaN during optimization.");
                }

                //Rho expresses the ratio between the actual reduction and the predicted reduction
                // (assuming the linerization was corect)
                const double dChi2 = i > 0 ? chi2Last-chi2(i) : 0;
                dchi2pred(i+1) = 0.5*dx.transpose() * (lambda(i)*dx + gradient);
                const double rho =   i > 0 ? dChi2/dchi2pred(i) : 0.5;
                 
                if ( rho > 0.75 )
                {
                    lambda(i+1) = std::max< double >( lambda(i) / _Ldown, double( 1e-7 ) );
                }else if (rho < 0.25){
                    lambda(i+1) = std::min< double >( lambda(i) * _Lup, double( 1e7 ) );
                }else{
                    lambda(i+1) = lambda(i);
                }
                if(rho > 0)
                {
                    xprev = x;
                    _computeResidual(x,r,W);
                    computeWeights(r,W);
                    chi2(i) = (r.transpose() * W.asDiagonal() * r);
                    _computeJacobian(x,J);
                    Eigen::MatrixXd H  = (J.transpose() * W.asDiagonal() * J);
                    chi2Last = chi2(i);

                }else{
                    x = xprev;
                }

                 SOLVER( INFO ) << "Iteration: " << i << 
                " chi2: " << chi2(i) << " dChi2: " << dChi2 << 
                " stepSize: " << stepSize(i) << " Total Weight: " << W.sum() <<
                " lambda: " << lambda(i) << " rho: " << rho;

            }
        }

        void LevenbergMarquardt::computeWeights(const Eigen::VectorXd& residuals, Eigen::VectorXd& weights)
        {
            // first order derivative of Tukey’s biweight loss function.
            // alternatively we could here assign weights based on the expected distribution of errors (e.g. t distribution)
            const auto t = residuals/algorithm::median(residuals);
            constexpr double kappa = 4.6851; //constant from paper
            constexpr double kappa2 = kappa*kappa;
            constexpr double kappa2_6 = kappa2/6;

            if (t.norm() <= kappa)
            {
                const auto t_k = t/kappa;
                for(int i = 0; i < weights.rows(); i++)
                {
                    weights(i) *= kappa2_6 * 1 - std::pow(( 1 - std::pow(t_k(i),2)),3);
                }
            
            }else{
                weights *= kappa2_6;
            }
              
        }
    }}