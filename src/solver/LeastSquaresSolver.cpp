#include "LeastSquaresSolver.h"
#include "utils/Log.h"
#include "utils/Exceptions.h"
#include "core/algorithm.h"

namespace pd{namespace vision{

    LeastSquaresSolver::LeastSquaresSolver(std::function<bool(const Eigen::MatrixXd&, Eigen::MatrixXd&,Eigen::MatrixXd&)> computeResidual,
            std::function<bool(const Eigen::MatrixXd&, Eigen::MatrixXd&)> computeJacobian,
            std::function<bool(const Eigen::MatrixXd&, Eigen::MatrixXd&)> updateX,
            int nObservations,
            int nParameters,
            double lambda0,
            double minStepSize,
            int maxIterations
            )
    :_computeResidual(computeResidual)
    ,_computeJacobian(computeJacobian)
    ,_updateX(updateX)
    ,_lambda0(lambda0)
    ,_maxIterations(maxIterations)
    ,_nObservations(nObservations)
    ,_nParameters(nParameters)
    ,_minStepSize(minStepSize)
    {
        Log::get("solver");
    }
    void LeastSquaresSolver::solve(Eigen::MatrixXd& x)
    {
        Eigen::VectorXd chiSquared(_maxIterations);
        chiSquared.setZero();
        Eigen::VectorXd lambda(_maxIterations);
        lambda.setConstant(_lambda0);
        Eigen::VectorXd stepSize(_maxIterations);
        stepSize.setZero();
        solve(x,chiSquared,lambda,stepSize);
    }

    double LeastSquaresSolver::computeChi2(const Eigen::MatrixXd& residuals, const Eigen::MatrixXd& weights)
    {
        double chiSquaredError     = 0.0;

        for ( int64_t i( 0 ); i < weights.size(); i++ )
        {
                chiSquaredError += residuals( i ) * residuals( i ) * weights( i );
        }
        return chiSquaredError;
    }

        void LeastSquaresSolver::solve(Eigen::MatrixXd &x, Eigen::VectorXd &chi2, Eigen::VectorXd &lambda, Eigen::VectorXd& stepSize) {
            Eigen::MatrixXd W(_nObservations, 1);
            W.setIdentity();
            Eigen::MatrixXd dx(_nParameters, 1);
            dx.setZero();
            Eigen::MatrixXd J(_nObservations, _nParameters);
            J.setZero();
            Eigen::MatrixXd residuals(_nObservations, 1);
            residuals.setConstant(std::numeric_limits <double >::max());
            _computeResidual(x,residuals,W);
            SOLVER( INFO ) << "Solving Problem for " << _nParameters << " parameters. With " << _nObservations << " observations.";

            chi2(0) = computeChi2(residuals,W);
            computeWeights(residuals,W);
            SOLVER( INFO ) << "0 > Residuals: " << residuals.norm() << " Chi2: " << chi2(0) << " Total Weight: " << W.sum();
            
            _computeJacobian(x,J);

            lambda(0) = (J.transpose()*W.asDiagonal()*J).norm();
            for(int i = 1; i < _maxIterations; i++)
            {
                // We want to solve dx = (JWJ + lambda * I)^(-1)*JWr
                // This can be solved with cholesky decomposition (Ax = b)
                // Where A = (JWJ + lambda * I), x = dx, b = JWr
                _computeJacobian(x,J);

                // For GN / LM we drop the second part of the Hessian
                Eigen::MatrixXd H  = (J.transpose() * W.asDiagonal() * J);
             
                // Lagrange multiplier steers magnitude and direction of the step
                // For lambda ~ 0 the update will be Gauss-Newton
                // For large lambda the update will be gradient
                H += lambda(i-1)*Eigen::MatrixXd::Identity(J.cols(),J.cols());
                
                SOLVER(DEBUG) << i << " > H.:\n" << H;

                
                const Eigen::MatrixXd gradient = J.transpose() * W.asDiagonal() * residuals;

                SOLVER(DEBUG) << i << " > Grad.:\n" << gradient.transpose() ;

                const Eigen::MatrixXd dx = H.ldlt().solve( gradient );

                SOLVER(DEBUG) << i <<" > x:\n" << x.transpose() ;
                SOLVER(DEBUG) << i <<" > dx:\n" << dx.transpose() ;

                Eigen::MatrixXd xi = x;
                _updateX(dx,xi);
                W.setZero();
                _computeResidual(xi,residuals,W);
                computeWeights(residuals,W);
                
                chi2(i) = computeChi2(residuals,W);
                SOLVER(DEBUG) << i << " > chi2: " << chi2(i);

                stepSize(i) = dx.norm();
                
                if ( chi2(i) < chi2(i-1) )
                {
                    lambda(i) = std::max< double >( lambda(i-1) / 9.0, double( 1e-7 ) );
                    x = xi;
                }else{
                    lambda(i) = std::min< double >( lambda(i-1) * 11.0, double( 1e7 ) );
                }
                SOLVER( INFO ) << "Iteration: " << i << " chi2: " << chi2(i) << " lambda: " << lambda(i) << " stepSize: " << stepSize(i) << " Total Weight: " << W.sum();
                
                 if ( stepSize(i) < _minStepSize )
                {
                    SOLVER( INFO ) << i << " > " << stepSize(i) << "/" << _minStepSize << " CONVERGED. ";
                    break;
                }

                if (!std::isfinite(stepSize(i)) || !std::isfinite(lambda(i)))
                {
                    throw pd::Exception(std::to_string(i) + "> NaN during optimization.");
                }

            }
        }

        void LeastSquaresSolver::computeWeights(const Eigen::MatrixXd& residuals, Eigen::MatrixXd& weights)
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

    GaussNewton::GaussNewton(std::function<bool(const Eigen::MatrixXd&, Eigen::MatrixXd&,Eigen::MatrixXd&)> computeResidual,
            std::function<bool(const Eigen::MatrixXd&, Eigen::MatrixXd&)> computeJacobian,
            std::function<bool(const Eigen::MatrixXd&, Eigen::MatrixXd&)> updateX,
            int nObservations,
            int nParameters,
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
    void GaussNewton::solve(Eigen::MatrixXd& x) const
    {
        Eigen::VectorXd chiSquared(_maxIterations);
        chiSquared.setZero();
        Eigen::VectorXd stepSize(_maxIterations);
        stepSize.setZero();
        solve(x,chiSquared,stepSize);
    }

    double GaussNewton::computeChi2(const Eigen::MatrixXd& residuals, const Eigen::MatrixXd& weights) const
    {
        double chiSquaredError     = 0.0;

        for ( int64_t i( 0 ); i < weights.size(); i++ )
        {
                chiSquaredError += residuals( i ) * residuals( i ) * weights( i );
        }
        return chiSquaredError;
    }

        void GaussNewton::solve(Eigen::MatrixXd &x, Eigen::VectorXd &chi2, Eigen::VectorXd& stepSize) const {
          
            SOLVER( INFO ) << "Solving Problem for " << _nParameters << " parameters. With " << _nObservations << " observations.";
            
            int iLast = 0;
            for(int i = 0; i < _maxIterations; i++)
            {
                Eigen::MatrixXd W = Eigen::MatrixXd::Zero(_nObservations, 1);
                Eigen::MatrixXd J = Eigen::MatrixXd::Zero(_nObservations, _nParameters);
                Eigen::MatrixXd residuals = Eigen::MatrixXd::Zero(_nObservations, 1);

                // We want to solve dx = (JWJ)^(-1)*JWr
                // This can be solved with cholesky decomposition (Ax = b)
                // Where A = (JWJ + lambda * I), x = dx, b = JWr

                _computeResidual(x,residuals,W);
                computeWeights(residuals,W);
                chi2(i) = computeChi2(residuals,W);

                _computeJacobian(x,J);
                // For GN / LM we drop the second part of the Hessian
                const Eigen::MatrixXd H  = (J.transpose() * W.asDiagonal() * J);
            
                SOLVER(DEBUG) << i << " > H.:\n" << H;
            
                const Eigen::MatrixXd gradient = J.transpose() * W.asDiagonal() * residuals;

                SOLVER(DEBUG) << i << " > Grad.:\n" << gradient.transpose() ;

                const Eigen::MatrixXd dx = _alpha * H.ldlt().solve( gradient );

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

        void GaussNewton::computeWeights(const Eigen::MatrixXd& residuals, Eigen::MatrixXd& weights) const
        {
                        
        }

       
    }}