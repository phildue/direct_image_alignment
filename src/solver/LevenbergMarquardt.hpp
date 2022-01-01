#include "utils/Log.h"
#include "utils/Exceptions.h"
#include "core/algorithm.h"

namespace pd{namespace vision{

    template<typename Problem,typename Loss>
    LevenbergMarquardt<Problem,Loss>::LevenbergMarquardt(
            int nObservations,
            int maxIterations,
            double minStepSize,
            double minGradient,
            double lambdaMin,
            double lambdaMax
            )
    :_maxIterations(maxIterations)
    ,_nObservations(nObservations)
    ,_minStepSize(minStepSize)
    ,_minGradient(minGradient)
    ,_lambdaMin(lambdaMin)
    ,_lambdaMax(lambdaMax)
    {
        Log::get("solver");
    }
    template<typename Problem,typename Loss>
    void LevenbergMarquardt<Problem,Loss>::solve(std::shared_ptr<Problem> problem) const
    {
        Eigen::VectorXd chiSquared(_maxIterations);
        chiSquared.setZero();
        Eigen::VectorXd chi2Predicted(_maxIterations);
        chi2Predicted.setZero();
       
        Eigen::VectorXd lambda(_maxIterations);
        lambda.setZero();
        Eigen::VectorXd stepSize(_maxIterations);
        stepSize.setZero();
        Eigen::Matrix<double, Eigen::Dynamic, Problem::nParameters> x(_maxIterations,Problem::nParameters);
        x.setZero();
        solve(problem,chiSquared,chi2Predicted,lambda,stepSize,x);
    }

    

    template<typename Problem,typename Loss>
    void LevenbergMarquardt<Problem,Loss>::solve(std::shared_ptr<Problem> problem, Eigen::VectorXd &chi2,Eigen::VectorXd &dchi2pred, Eigen::VectorXd &lambda, Eigen::VectorXd& stepSize, Mmxn& x) const{
        SOLVER( INFO ) << "Solving Problem for " << Problem::nParameters << " parameters. With " << _nObservations << " observations.";

    
        Eigen::VectorXd W = Eigen::VectorXd::Zero(_nObservations);
        Mmxn J = Eigen::MatrixXd::Zero(_nObservations, Problem::nParameters);
        Eigen::VectorXd r = Eigen::VectorXd::Zero(_nObservations);
        /**
         * 2 IxydWp^T*(Iwxp + IxydWp * dp - T) = 0
         * IxydWp^T*IxydWp * dp + IxydWp^T( Iwxp - T) = 0
         * IxydWp^T*( Iwxp - T) = -IxydWp^T*IxydWp * dp
         * -IxydWp^T( Iwxp - T) = IxydWp^T*IxydWp * dp
         * IxydWp^T( T-Iwxp ) = IxydWp^T*IxydWp * dp 
         * J^T*r = H*dp
         * dp =H^(-1)*J^T*r
         * **/
        // We want to solve dx = (H)^(-1)*JWr
        // This can be solved with cholesky decomposition (Ax = b)
        // Where A = (JWJ + lambda * I), x = dx, b = JWr

        problem->computeResidual(r);
        Loss::computeWeights(r,W);
        chi2(0) = (r.transpose() * W.asDiagonal() * r);
        problem->computeJacobian(J);
        // For GN / LM we drop the second part of the Hessian
        Eigen::MatrixXd H  = (J.transpose() * W.asDiagonal() * J);
        double chi2Last;
        for(int i = 0; i < _maxIterations -1; i++)
        {
            x.row(i) = problem->x();

            if ( i == 0)
            {
                lambda(i) = std::min< double >( H.norm(), double( _lambdaMax ) );
            }
            // Lagrange multiplier steers magnitude and direction of the step
            // For lambda ~ 0 the update will be Gauss-Newton
            // For large lambda the update will be gradient descent
            H.noalias() += lambda(i)*Eigen::MatrixXd::Identity(J.cols(),J.cols());
            
            SOLVER(DEBUG) << i << " > H.:\n" << H;
        
            const Eigen::VectorXd gradient = J.transpose() * W.asDiagonal() * r;

            SOLVER(DEBUG) << i << " > Grad.:\n" << gradient.transpose() ;

            const Eigen::VectorXd dx = (H.ldlt().solve( gradient ));
            //const Eigen::VectorXd dx = H.inverse()* gradient ;

            SOLVER(DEBUG) << i <<" > x:\n" << problem->x().transpose() ;
            SOLVER(DEBUG) << i <<" > dx:\n" << dx.transpose() ;
            stepSize(i) = dx.norm();

            //Rho expresses the ratio between the actual reduction and the predicted reduction
            // (assuming the linerization was corect)
            const double dChi2 = i > 0 ? chi2Last-chi2(i) : 0;
            dchi2pred(i+1) = 0.5*dx.transpose() * (lambda(i)*dx + gradient);
            const double rho =   i > 0 ? dChi2/dchi2pred(i) : 0.5;
                
            if ( rho > 0.75 )
            {
                lambda(i+1) = std::max< double >( lambda(i) / _Ldown, double( _lambdaMin ) );
            }else if (rho < 0.25){
                lambda(i+1) = std::min< double >( lambda(i) * _Lup, double( _lambdaMax ) );
            }else{
                lambda(i+1) = lambda(i);
            }
            if(rho > 0)
            {
                problem->updateX(dx);
                problem->computeResidual(r);
                Loss::computeWeights(r,W);
                chi2(i) = (r.transpose() * W.asDiagonal() * r);
                problem->computeJacobian(J);
                H  = (J.transpose() * W.asDiagonal() * J);
                chi2Last = chi2(i);

            }
            
            SOLVER( INFO ) << "Iteration: " << i << 
            " chi2: " << chi2(i) << " dChi2: " << dChi2 << 
            " stepSize: " << stepSize(i) << " Total Weight: " << W.sum() <<
            " lambda: " << lambda(i) << " rho: " << rho;
            if ( stepSize(i) < _minStepSize /*|| std::abs(gradient.maxCoeff()) < _minGradient*/)
            {
                SOLVER( INFO ) << i << " > Stepsize: " << stepSize(i) << "/" << _minStepSize << 
                " Gradient: " << gradient.maxCoeff() << "/" << _minGradient << " CONVERGED. ";
                break;
            }

            if (!std::isfinite(stepSize(i)))
            {
                throw pd::Exception(std::to_string(i) + "> NaN during optimization.");
            }
            }
        }

    
}}