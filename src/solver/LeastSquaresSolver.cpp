#include "LeastSquaresSolver.h"
#include "utils/Log.h"

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

    }
    void LeastSquaresSolver::solve(Eigen::MatrixXd& x)
    {
        Eigen::VectorXd chiSquared(_maxIterations);
        Eigen::VectorXd lambda(_maxIterations);
        solve(x,chiSquared,lambda);
    }

    double LeastSquaresSolver::computeChi2(const Eigen::MatrixXd& residuals, const Eigen::MatrixXd& weights)
    {
        return (residuals.array() * residuals.array() * weights.array()).sum();
    }

        void LeastSquaresSolver::solve(Eigen::MatrixXd &x, Eigen::VectorXd &chi2, Eigen::VectorXd &lambda) {
            Eigen::MatrixXd W(_nObservations, 1);
            Eigen::MatrixXd dx(_nParameters, 1);
            Eigen::MatrixXd J(_nObservations, _nParameters);
            Eigen::MatrixXd residuals(_nObservations, 1);
            _computeResidual(x,residuals,W);

            chi2(0) = computeChi2(residuals,W);
            VLOG(4) << "0 > Residuals: " << residuals.norm() << " Chi2: " << chi2(0) << " #Observations: " << W.sum();

            lambda(0) = _lambda0;
            for(int i = 1; i < _maxIterations; i++)
            {
                _computeJacobian(x,J);

                Eigen::MatrixXd H  = J.transpose() * W.asDiagonal() * J;
                VLOG(4) << i << " > H.:\n" << H;

                const Eigen::MatrixXd JWJ = H.diagonal();

                for ( int ih = 0; ih < _nParameters; ih++ )
                {
                    H( ih, ih ) += lambda(i-1) * JWJ( ih );
                }

                const Eigen::MatrixXd gradient = J.transpose() * W.asDiagonal() * residuals;

                VLOG(4) << i << " > Grad.:\n" << gradient.transpose() ;

                dx.noalias() = H.ldlt().solve( gradient );

                VLOG(4) << i <<" > x:\n" << x.transpose() ;
                VLOG(4) << i <<" > dx:\n" << dx.transpose() ;

                Eigen::MatrixXd xi = x;
                _updateX(dx,xi);
                _computeResidual(xi,residuals,W);

                chi2(i) = computeChi2(residuals,W);
                VLOG(4) << i << " > chi2: " << chi2(i);

                const double stepSize = (dx.array() * dx.array()).sum();
                VLOG(4) << i << " > stepSize: " << stepSize;

                if ( stepSize < _minStepSize )
                {
                    VLOG(4) << i << " > CONVERGED. ";
                    break;
                }

                if ( chi2(i) < chi2(i-1) )
                {
                    lambda(i) = std::max< double >( lambda(i-1) / 9.0, double( 1e-7 ) );
                    x = xi;
                }else{
                    lambda(i) = std::min< double >( lambda(i-1) * 11.0, double( 1e7 ) );
                }
            }
        }

    }}