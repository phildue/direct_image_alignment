#include <utils/utils.h>
#include "Scaler.h"
using namespace pd::vision;
namespace pd::vslam::solver
{
    void MedianScaler::compute(const vision::VecXd& r)
    {
        /*
        std::vector<double> rs();
        rs.reserve(r.rows());
        for (int i = 0; i < r.rows(); i++)
        {
            algorithm::insertionSort(rs,r(i));
        }*/

        _median = algorithm::median(r,false);
        _std = std::sqrt((r.array() - _median).array().abs().sum()/(r.rows() - 1));
        LOG_PLT("MedianScaler") << std::make_shared<vis::Histogram>(r,"ErrorDistribution",30);
    }

    VecXd MedianScaler::scale(const VecXd& r) { 
        
        return (r.array() - _median)/_std; 
    }

    void MeanScaler::compute(const vision::VecXd& r)
    {

        _mean = r.mean();
        _std = std::sqrt((r.array() - _mean).array().abs().sum()/(r.rows() - 1));
        LOG_PLT("MedianScaler") << std::make_shared<vis::Histogram>(r,"ErrorDistribution",30);
    }

    VecXd MeanScaler::scale(const VecXd& r) { 
        
        return (r.array() - _mean)/_std; 
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
                sum += r(i)*r(i) * (_v+1)/(_v+r(i)/_sigma);
            }
            const double sigma_i = std::sqrt(sum/(double)r.rows());
            stepSize = std::abs(_sigma - sigma_i);
            _sigma = sigma_i;
        }

        return r / _sigma;

     }

    
    
    
} // namespace pd::vslam::solver
