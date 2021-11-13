#ifndef VSLAM_LOSS_H__
#define VSLAM_LOSS_H__

#include <Eigen/Dense>

namespace pd{namespace vision{

class TukeyLoss
{
        public:
        static void computeWeights(const Eigen::VectorXd& residuals, Eigen::VectorXd& weights);
};        

}}
#endif