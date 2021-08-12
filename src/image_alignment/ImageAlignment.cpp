#include <utility>
#include <iostream>

#include <ceres/ceres.h>
#include <ceres/cubic_interpolation.h>
#include <ceres/rotation.h>

#include "utils/Log.h"
#include "ImageAlignment.h"
#include "core/Camera.h"
#include "core/algorithm.h"
#include "core/Point3D.h"
#include "core/Feature2D.h"
#include "utils/Exceptions.h"
#include "utils/utils.h"
#include "PhotometricLoss.h"
#include "LocalParameterizationSE3.h"

namespace pd{ namespace vision{

    class PhotometricError{
        const Feature2D::ShConstPtr _ftRef;
        const Frame::ShConstPtr _frameTarget;
        const int _patchSize;
        const int _patchSizeHalf;
        const int _level;
        const double _scale;
        const double _cx,_cy,_f;
        Eigen::MatrixXd _patchRef;
        Eigen::Vector3d  _p3d;
        ceres::Grid2D<double> _img;
        ceres::BiCubicInterpolator<ceres::Grid2D<double>> _interpolator;
    public:
        PhotometricError(Feature2D::ShConstPtr ftRef,
                         Frame::ShConstPtr frameTarget,
                         uint32_t patchSize,
                         uint32_t level)
                         : _ftRef(ftRef)
                         , _frameTarget(std::move(frameTarget))
                         , _patchSize(patchSize)
                         , _patchSizeHalf(patchSize/2)
                         , _level(level)
                         , _scale(1.0/(1U<<level))
                         , _f(ftRef->frame()->camera()->focalLength())
                , _cx(ftRef->frame()->camera()->principalPoint().x())
                , _cy(ftRef->frame()->camera()->principalPoint().y())
                , _img((double*)_frameTarget->grayImage().data(),0,_frameTarget->height(),0,_frameTarget->width())
                , _interpolator(_img)
        {
            if(!ftRef->point())
            {
                throw pd::Exception("Feature [" + std::to_string(ftRef->id()) + "] does not have corresponding 3D point.");
            }
            if(!ftRef->frame()->isVisible(ftRef->position(),_patchSize,level))
            {
                throw pd::Exception("Feature [" + std::to_string(ftRef->id()) + "] is not fully visible in reference frame.");

            }
            _p3d = ftRef->point()->position();

            const auto frameRef = ftRef->frame();
            const auto mat = frameRef->grayImage(level);
            const auto pImg = _ftRef->position();
            _patchRef.resize (_patchSize,_patchSize);
            int idxPixel = 0;
            for (int i = 0; i < _patchSize; i++)
            {
                for (int j = 0; j < _patchSize; j++) {
                    _patchRef(i, j) = algorithm::bilinearInterpolation(mat,
                                                                  (pImg.y() - _patchSizeHalf + i) * _scale,
                                                                  (pImg.x() - _patchSizeHalf + j) * _scale);
                }
            }
        }

        template <typename T>
        bool operator()(const T* const sTwa, T* sResiduals) const {

            const Eigen::Map<const Sophus::SE3<T> > T_wa(sTwa);
            Eigen::Map<Eigen::Matrix<T,49,1> > residuals(sResiduals);

            const Eigen::Matrix<T,3,1> p3dCcs = T_wa * _p3d.cast<T>();
            //TODO check z > 0?

            Eigen::Matrix<T,3,1> pProj = _frameTarget->camera()->K() * p3dCcs;
            pProj /= pProj.z();

            //TODO check image boundaries?

            int idxPixel(0);

            for (int i = 0; i < _patchSizeHalf; i ++)
            {
                for (int j = 0; j < _patchSizeHalf; j ++)
                {
                    T imgX = (pProj.x() + ((double)i -(double)_patchSizeHalf))*_scale;
                    T imgY = (pProj.y() + ((double)j -(double)_patchSizeHalf))*_scale;
                    T f;
                    _interpolator.Evaluate(imgX, imgY, &f);

                    residuals(idxPixel++,0) = f - (T)_patchRef(i,j);
                }
            }

            return true;


        }

        };



    Sophus::SE3d ImageAlignment::align(Frame::ShConstPtr referenceFrame, Frame::ShConstPtr targetFrame) const{

        Sophus::SE3d pose = targetFrame->pose();
        for (int level = _levelMax-1; level >= _levelMin; --level)
        {

            //const Eigen::Matrix<double,6,1> v6d = pose.log();
            //double poseArray[6] = {v6d(0),v6d(1),v6d(2),v6d(3),v6d(4),v6d(5)};

            VLOG(4) << "IA init: " << " Level: " << level  << " #Features: " << referenceFrame->features().size();
            ceres::Problem problem;
            for ( int idxF = 0; idxF < referenceFrame->features().size(); idxF++)
            {
                const auto& f = referenceFrame->features()[idxF];
                if ( f->point() )
                {
                    if ( referenceFrame->isVisible(f->position(),_patchSize, level))
                    {
                    //    auto cost = new PhotometricLoss(f,targetFrame,_patchSize,level);
                        auto cost = new ceres::AutoDiffCostFunction<PhotometricError,Sophus::SE3d::DoF,
                                Sophus::SE3d::num_parameters>(new PhotometricError(f,targetFrame,_patchSize,level));
                        problem.AddParameterBlock(pose.data(),Sophus::SE3d::num_parameters,new LocalParameterizationSE3());
                        problem.AddResidualBlock(cost,new ceres::HuberLoss(10.0),pose.data());

                    }
                }
            }

            ceres::Solver::Options options{};
            options.linear_solver_type = ceres::DENSE_SCHUR;
            options.minimizer_progress_to_stdout = true;

            ceres::Solver::Summary summary{};
            VLOG(4) << "Setup IA with #Parameters: " << problem.NumParameters() << ", #Residuals: " << problem.NumResiduals();
            ceres::Solve(options, &problem, &summary);

            //TODO: set output pose again
            if (VLOG_IS_ON(4))
            {
                VLOG(4) << summary.FullReport();

            }else{
                VLOG(3) << summary.BriefReport();

            }

        }


        return pose;
    }

    ImageAlignment::ImageAlignment(uint32_t levelMax, uint32_t levelMin, uint32_t patchSize)
    : _levelMax(levelMax)
    , _levelMin(levelMin)
    , _patchSize(patchSize)
    {}
}
}
