#include "Scaler.h"
using namespace pd::vision;
namespace pd::vslam::solver
{
    VecXd MedianScaler::scale(const VecXd& r) { 
        std::vector<double> rs;
        rs.reserve(r.rows());
        for (int i = 0; i < r.rows(); i++)
        {
            if(std::isfinite(r(i)))
            {
                algorithm::insertionSort(rs,r(i));
            }
        }
        const double med = algorithm::median(rs,true);
        const Eigen::Map<Eigen::VectorXd> rv(rs.data(),rs.size());
        const double std = (rv.array() - med).array().abs().sum()/(rv.rows() - 1);

        return (r.array() - med)/std; 
        
    }

    VecXd ScalerTDistribution::scale(const VecXd& r) { 
        double stepSize = std::numeric_limits<double>::max();

        for(size_t iter = 0; 
        iter < _maxIterations && stepSize < _minStepSize;
         iter++)
        {
            double sum = 0.0;
            for (int i = 0; i < r.rows(); i++)
            {
                if(std::isfinite(r(i)))
                {
                    sum += r(i)*r(i) * (_v+1)/(_v+r(i)/_sigma);
                }
            }
            const double sigma_i = std::sqrt(sum/(double)r.rows());
            stepSize = std::abs(_sigma - sigma_i);
            _sigma = sigma_i;
        }

        return r / _sigma;

     }

    
    
    
} // namespace pd::vslam::solver
