#include "utils/visuals.h"
#include "utils/Log.h"
#include "utils/Exceptions.h"

namespace pd{namespace vision{
    #define LOG_PLOT_LM(name) Log::getPlotLog(name,Level::Debug)

    template<typename Problem>
    LevenbergMarquardt<Problem>::LevenbergMarquardt(
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
    ,_minReduction(minStepSize)
    ,_lambdaMin(lambdaMin)
    ,_lambdaMax(lambdaMax)
    {
        Log::get("solver");
    }
    template<typename Problem>
    void LevenbergMarquardt<Problem>::solve(std::shared_ptr<Problem> problem) 
    {
        Eigen::VectorXd chiSquared = Eigen::VectorXd::Zero(_maxIterations);
        Eigen::VectorXd chi2Predicted = Eigen::VectorXd::Zero(_maxIterations);
       
        Eigen::VectorXd lambda = Eigen::VectorXd::Zero(_maxIterations);
        Eigen::VectorXd stepSize = Eigen::VectorXd::Zero(_maxIterations);

        Eigen::Matrix<double, Eigen::Dynamic, Problem::nParameters> x(_maxIterations,Problem::nParameters);
        solve(problem,chiSquared,chi2Predicted,lambda,stepSize,x);
    }

    

    template<typename Problem>
    void LevenbergMarquardt<Problem>::solve(std::shared_ptr<Problem> problem, Eigen::VectorXd &chi2,Eigen::VectorXd &dChi2pred, Eigen::VectorXd &lambda, Eigen::VectorXd& stepSize, Mmxn& x) const{
        SOLVER( INFO ) << "Solving Problem for " << Problem::nParameters << " parameters. With " << _nObservations << " observations.";

        Eigen::VectorXd rho = Eigen::VectorXd::Zero(_maxIterations);
        Eigen::VectorXd dChi2 = Eigen::VectorXd::Zero(_maxIterations);

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
        // Lagrange multiplier (damping factor) steers magnitude and direction of the step
        // For lambda ~ 0 the update will be Gauss-Newton
        // For large lambda the update will be gradient descent
        
        Eigen::VectorXd r,w;
        problem->computeResidual(r,w);

        Mmxn J = Eigen::MatrixXd::Zero(r.rows(), Problem::nParameters);
   
        chi2(0) = r.transpose() * w.asDiagonal() * r;
        problem->computeJacobian(J);
        // For GN / LM we drop the second part of the Hessian
        Eigen::MatrixXd H  = (J.transpose() * w.asDiagonal() * J);
        lambda(0) = std::max<double>(H.norm(),_lambdaMax);

        int i = 0;
        for(; i < _maxIterations; i++)
        {
            x.row(i) = problem->x();

            Eigen::Matrix<double,Problem::nParameters,Problem::nParameters> dampingMat;
            for (int j = 0; j < Problem::nParameters; j++)
            {
                dampingMat(j,j) = lambda(i);//H(j,j);
            }
            Eigen::Matrix<double,Problem::nParameters,Problem::nParameters> Hprev = H;
            H.noalias() += dampingMat;
            
            SOLVER(DEBUG) << i << " > H.:\n" << H;
        
            const Eigen::VectorXd gradient = J.transpose() * w.asDiagonal() * r;

            SOLVER(DEBUG) << i << " > Grad.:\n" << gradient.transpose() ;

            const Eigen::VectorXd dx = (H.ldlt().solve( gradient ));

            SOLVER(DEBUG) << i <<" > x:\n" << problem->x().transpose() ;
            SOLVER(DEBUG) << i <<" > dx:\n" << dx.transpose() ;
            stepSize(i) = dx.norm();
            
            problem->updateX(dx);
            const Eigen::VectorXd rPrev = r;
            const Eigen::VectorXd wPrev = w;
            problem->computeResidual(r,w);

            chi2(i+1) = r.transpose() * w.asDiagonal() * r;

            // Rho expresses the ratio between the actual reduction and the predicted reduction
            // (assuming the linearization was correct)
            dChi2pred(i) = dx.transpose() * (dampingMat.diagonal() * dx + gradient);
            dChi2(i) = chi2(i)-chi2(i+1);
            rho(i) =   dChi2(i)/dChi2pred(i);

            SOLVER( INFO ) << "Iteration: " << i << 
            " chi2: " << chi2(i) << " dChi2: " << dChi2(i) << 
            " stepSize: " << stepSize(i) << 
            " lambda: " << lambda(i) << " rho: " << rho(i);
            if (!std::isfinite(stepSize(i)))
            {
                throw pd::Exception(std::to_string(i) + "> NaN during optimization.");
            }
        
                
            if ( rho(i) > 0.75 )
            {
                lambda(i+1) = std::max< double >( lambda(i) / _Ldown, double( _lambdaMin ) );

            }else if (0.25 <= rho(i) && rho(i) <= 0.75){

                lambda(i+1) = lambda(i);
            }
            else if (rho(i) < 0.25){

                lambda(i+1) = std::min< double >( lambda(i) * _Lup, double( _lambdaMax ) );
            }

            if ( rho(i) < 0 && lambda(i) < _lambdaMax)
            {
                //If the step was not positive (actual reduction) we revert the changes to x
                //Since only lambda has changed we keep J and H as before
                problem->updateX(-dx);
                r = rPrev;
                w = wPrev;
                chi2(i+1) = chi2(i);
                H = Hprev;
                stepSize(i) = stepSize(i-1);
                continue;
            }

            if ( stepSize(i) < _minStepSize || std::abs(gradient.maxCoeff()) < _minGradient || std::abs(dChi2(i)) < _minReduction)
            {
                SOLVER( INFO ) << i << " > Stepsize: " << stepSize(i) << "/" << _minStepSize << 
                " Gradient: " << gradient.maxCoeff() << "/" << _minGradient << " CONVERGED. ";
                break;
            }

            //If the step was positive, we keep the new x
            //Since x has changed we need to recompute J and H
            problem->computeJacobian(J);
            H  = (J.transpose() * w.asDiagonal() * J);
           
        }

        LOG_PLOT_LM("SolverLM") << std::make_shared<vis::PlotLevenbergMarquardt>(i,chi2,dChi2,dChi2pred,lambda,stepSize,rho);
    }

    
}}