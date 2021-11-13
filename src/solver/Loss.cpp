#include "Loss.h"
#include "core/algorithm.h"
namespace pd{ namespace vision{

    void TukeyLoss::computeWeights(const Eigen::VectorXd& residuals, Eigen::VectorXd& weights)
    {
        // first order derivative of Tukey’s biweight loss function.
        // alternatively we could here assign weights based on the expected distribution of errors (e.g. t distribution)
        const auto t = residuals/algorithm::median(residuals);
        constexpr double kappa = 4.6851; //constant from paper
        constexpr double kappa2 = kappa*kappa;
        constexpr double kappa2_6 = kappa2/6;
        weights.setZero();
        if (t.norm() <= kappa)
        {
            const auto t_k = t/kappa;
            for(int i = 0; i < weights.rows(); i++)
            {
                weights(i) = kappa2_6 * 1 - std::pow(( 1 - std::pow(t_k(i),2)),3);
            }
        
        }else{
            weights.setConstant(kappa2_6);
        }
    }
}}