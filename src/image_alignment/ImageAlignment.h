#ifndef DIRECT_IMAGE_IMAGE_ALIGNMENT_H__
#define DIRECT_IMAGE_IMAGE_ALIGNMENT_H__


#include <sophus/se3.hpp>
#include <ceres/ceres.h>

#include "core/Frame.h"

namespace pd{namespace vision{

    template<int patchSize>
    class ImageAlignment
    {
    public:
        explicit ImageAlignment(uint32_t levelMax, uint32_t levelMin);
        virtual void align(Frame::ShConstPtr referenceFrame, Frame::ShConstPtr targetFrame) const;

        struct Cost : ceres::SizedCostFunction<patchSize*patchSize,Sophus::SE3d::DoF>{
            Eigen::Matrix<double, Sophus::SE3d::DoF,Eigen::Dynamic,Eigen::ColMajor> _jacobian;
            Eigen::MatrixXd _patchRef;
            const Eigen::Vector3d  _p3d;
            const Feature2D::ShConstPtr _ftRef;
            const Frame::ShConstPtr  _frameTarget;
            const std::uint32_t _patchSize;
            const std::uint32_t  _patchArea;
            const std::uint32_t  _patchSizeHalf;
            const std::uint32_t  _level;
            const double _scale;
            Cost (Feature2D::ShConstPtr ftRef,
                             Frame::ShConstPtr frameTarget,
                             std::uint32_t level
            );
            bool Evaluate(double const* const* parameters,double* residuals,double** jacobians) const ;
        };

    protected:
        const int _levelMax,_levelMin;



    };



}}

#include "ImageAlignment.hpp"
#include "ImageAlignmentAutoDiff.hpp"
#include "PhotometricLoss.hpp"
#endif //IMAGE_ALIGNMENT_H__
