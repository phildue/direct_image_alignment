#include <utility>

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

namespace pd{
namespace vision{

    struct PhotometricLoss : ceres::SizedCostFunction<ceres::DYNAMIC,6>{
        // gray value of original tag point
        // 3d point in world coordinate
        Eigen::MatrixXd _jacobian;
        Eigen::MatrixXd _patchRef;
        const Eigen::Vector3d  _p3d;
        const Feature2D::ShConstPtr _ftRef;
        const Frame::ShConstPtr  _frameTarget;
        const uint32_t _patchSize;
        const uint32_t  _patchArea;
        const uint32_t  _patchSizeHalf;
        const uint32_t  _level;
        const double _scale;
        PhotometricLoss (Feature2D::ShConstPtr ftRef,
                         Frame::ShConstPtr frameTarget,
                         uint32_t patchSize,
                         uint32_t level
                         )
                         : _ftRef(ftRef)
                         , _frameTarget(frameTarget)
                         , _p3d(ftRef->point()->position())
                         , _patchSize(patchSize)
                         , _patchArea(patchSize*patchSize)
                         , _level(level)
                         , _scale(1.0/(1U<<level))
                         , _patchSizeHalf(patchSize/2){

            if(!ftRef->point())
            {
                throw pd::Exception("Feature [" + std::to_string(ftRef->id()) + "] does not have corresponding 3D point.");
            }
            if(!ftRef->frame()->isVisible(ftRef->position(),_patchSize,level))
            {
                throw pd::Exception("Feature [" + std::to_string(ftRef->id()) + "] is not fully visible in reference frame.");

            }

            const auto frameRef = ftRef->frame();
            const auto mat = frameRef->grayImage(level);
            const auto pCamera = frameRef->pose() * _p3d;
            const auto jacobianFeature = frameRef->camera()->J_xyz2uv(pCamera);
            const auto pImg = _ftRef->position();
            _jacobian.resize(6,patchSize*patchSize);
            _patchRef.resize (_patchSize,_patchSize);
            int idxPixel = 0;
            for (int i = 0; i < _patchSize; i++)
            {
                for (int j = 0; j < _patchSize; j++)
                {
                    _patchRef(i,j) = algorithm::bilinearInterpolation(mat,
                                                                (pImg.y() - _patchSizeHalf + i) *_scale,
                                                                (pImg.x() - _patchSizeHalf + j) *_scale);

                    const double fx1 = algorithm::bilinearInterpolation(mat,
                                                                   (pImg.y() - _patchSizeHalf + i) *_scale,
                                                                   (pImg.x() - _patchSizeHalf + j + 1) *_scale);
                    const double fx0 = algorithm::bilinearInterpolation(mat,
                                                                   (pImg.y() - _patchSizeHalf + i ) *_scale,
                                                                   (pImg.x() - _patchSizeHalf + j ) *_scale);
                    const double fy1 = algorithm::bilinearInterpolation(mat,
                                                                   (pImg.y() - _patchSizeHalf + i + 1) *_scale,
                                                                   (pImg.x() - _patchSizeHalf + j) *_scale);
                    const double fy0 = algorithm::bilinearInterpolation(mat,
                                                                   (pImg.y() - _patchSizeHalf + i ) *_scale,
                                                                   (pImg.x() - _patchSizeHalf + j ) *_scale);

                    double dx = 0.5 * (fx1 - fx0);
                    double dy = 0.5 * (fy1 - fy0);

                    _jacobian.col(idxPixel) = (dx*jacobianFeature.row(0) + dy*jacobianFeature.row(1))*(frameRef->camera()->focalLength() / _scale);

                }
                idxPixel++;
            }

            VLOG(5) << "Photometric Loss for feature [ " << ftRef->id() << " ]:"
                                                                           << " \n W = " << _patchRef
                                                                           << " \n J = " << _jacobian;
            this->set_num_residuals(_patchArea);

        }

        bool Evaluate(double const* const* parameters,double* residuals,double** jacobians) const {

            Sophus::SE3d pose;
            const auto pCamera = pose * _p3d;
            const auto pImg = _frameTarget->camera2image(pCamera);

            if ( _frameTarget->isVisible(pImg,_patchSizeHalf,_level))
            {
                int idxPixel = 0;
                for (int i = 0; i < _patchSize; i++)
                {
                    for (int j = 0; j < _patchSize; j++)
                    {
                        auto idxRow = static_cast<std::uint32_t > ( i * _scale);
                        auto idxCol = static_cast<std::uint32_t > ( j * _scale);
                        double grayTarget =     algorithm::bilinearInterpolation(_frameTarget->grayImage(_level),
                                                                            (pImg.y() - _patchSizeHalf + i) *_scale,
                                                                            (pImg.x() - _patchSizeHalf + j) *_scale);
                        residuals[idxPixel] = (_patchRef(i,j) - grayTarget);

                        idxPixel++;
                    }

                }
                jacobians[0] = (double*)_jacobian.data();
                return true;
            }else{
                return false;

            }
    }
    };

    class PhotometricError{
        const Feature2D::ShConstPtr _ftRef;
        const Frame::ShConstPtr _frameTarget;
        const uint32_t _patchSize;
        const uint32_t _patchSizeHalf;
        const uint32_t _level;
        const double _scale;
        const double _cx,_cy,_f;
        Eigen::MatrixXd _patchRef;
        double  _p3d[3];
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
            memcpy(_p3d,ftRef->point()->position().data(),3*sizeof(double));

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
        bool operator()(const T* const pose, T* residuals) const {


            T pCamera[3];
            ceres::AngleAxisRotatePoint(pose, (T*)_p3d, pCamera);
            pCamera[0] += pose[3];
            pCamera[1] += pose[4];
            pCamera[2] += pose[5];

            T u = _f * pCamera[0] / pCamera[2] + _cx;
            T v = _f * pCamera[1] / pCamera[2] + _cy;


            computeResidual(u,v,residuals);
            return true;


        }
        template <typename T>
        void computeResidual(T u, T v, T* residuals) const
        {
            int idxPixel = 0;

            for (int i = -_patchSizeHalf; i < _patchSizeHalf; i++)
            {
                for (int j = -_patchSizeHalf; j < _patchSizeHalf; j++)
                {
                    auto idxRow = static_cast<std::uint32_t > ( i * _scale);
                    auto idxCol = static_cast<std::uint32_t > ( j * _scale);
                    auto imgX = (u + (T)i)*_scale;
                    auto imgY = (v + (T)j)*_scale;
                    T f;
                    _interpolator.Evaluate(imgX, imgY, &f);

                    residuals[idxPixel] = f - (T)_patchRef(i,j);
                }
            }
        }
        };



    Sophus::SE3d ImageAlignment::align(Frame::ShConstPtr referenceFrame, Frame::ShConstPtr targetFrame) const{

        Sophus::SE3d pose = targetFrame->pose();
        for (int level = _levelMax-1; level >= _levelMin; --level)
        {

            const Eigen::Matrix<double,6,1> v6d = pose.log();
            double poseArray[6] = {v6d(0),v6d(1),v6d(2),v6d(3),v6d(4),v6d(5)};

            VLOG(4) << "IA init: " << " Level: " << level << " Pose: " << v6d << " #Features: " << referenceFrame->features().size();
            ceres::Problem problem;

            for ( int idxF = 0; idxF < referenceFrame->features().size(); idxF++)
            {
                const auto& f = referenceFrame->features()[idxF];
                if ( f->point() )
                {
                    if ( referenceFrame->isVisible(f->position(),level))
                    {
                        auto cost = new PhotometricLoss(f,targetFrame,_patchSize,level);
                    //    auto cost = new ceres::AutoDiffCostFunction<PhotometricError,49,6>(new PhotometricError(f,targetFrame,_patchSize,level));
                        problem.AddResidualBlock(cost,new ceres::HuberLoss(10.0),&poseArray[0]);

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
    : _levelMax(levelMax),
    _levelMin(levelMin),
    _patchSize(patchSize){

    }
}
}
