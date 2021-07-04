#include <utility>

#include <iostream>
#include <ceres/ceres.h>
#include <ceres/cubic_interpolation.h>
#include <ceres/rotation.h>

#include "ImageAlignment.h"
#include "Camera.h"
#include "math.h"
#include "Point3D.h"
#include "Feature2D.h"

namespace pd{
namespace vision{

    struct PhotometricLoss : ceres::CostFunction{
        // gray value of original tag point
        // 3d point in world coordinate
        Eigen::MatrixXd _jacobian;
        std::vector<Eigen::MatrixXd> _patchesRef;
        std::vector<Feature2D::ShConstPtr> _featuresWithPoints;
        Frame::ShConstPtr  _frameRef,_frameTarget;
        const uint32_t _patchSize;
        const uint32_t  _patchArea;
        const uint32_t  _patchSizeHalf;
        const uint32_t  _level;
        const double _scale;
        PhotometricLoss (Frame::ShConstPtr frameRef,
                         Frame::ShConstPtr frameTarget,
                         uint32_t patchSize,
                         uint32_t level
                         ):
                        _frameRef(std::move(frameRef)),
                        _frameTarget(std::move(frameTarget)),
                        _patchSize(patchSize),
                        _patchArea(patchSize*patchSize),
                        _level(level),
                        _scale(1.0/(1U<<level)),
                        _patchSizeHalf(patchSize/2){

            const auto nObservedPoints = frameRef->nObservedPoints();
            _featuresWithPoints.reserve(nObservedPoints);
            _patchesRef.resize(nObservedPoints);
            _jacobian = Eigen::MatrixXd(nObservedPoints,patchSize*patchSize*6);
            const auto mat = _frameRef->grayImage(level);
            for ( int idxF = 0; idxF < frameRef->features().size(); idxF++)
            {
                const auto& f = frameRef->features()[idxF];
                if ( f->point() )
                {
                    _featuresWithPoints.push_back(f);
                    const auto idxObservation = _featuresWithPoints.size();
                    const auto& p3d = f->point();

                    if ( frameRef->isVisible(f->position(),level))
                    {
                        const auto pCamera = frameRef->pose() * f->point()->position();
                        const auto jacobianFeature = frameRef->camera()->J_xyz2uv(pCamera);
                        Eigen::MatrixXd patchRef(_patchSize,_patchSize);
                        const auto pImg = _frameRef->world2image(f->point()->position());
                        int idxPixel = 0;
                        for (int i = 0; i < _patchSize; i++)
                        {
                            for (int j = 0; j < _patchSize; j++)
                            {
                                patchRef(i,j) = math::bilinearInterpolation(mat,
                                                                            (pImg.y() - _patchSizeHalf + i) *_scale,
                                                                            (pImg.x() - _patchSizeHalf + j) *_scale);

                                const double fx1 = math::bilinearInterpolation(mat,
                                                                               (pImg.y() - _patchSizeHalf + i) *_scale,
                                                                               (pImg.x() - _patchSizeHalf + j + 1) *_scale);
                                const double fx0 = math::bilinearInterpolation(mat,
                                                                               (pImg.y() - _patchSizeHalf + i ) *_scale,
                                                                               (pImg.x() - _patchSizeHalf + j ) *_scale);
                                const double fy1 = math::bilinearInterpolation(mat,
                                                                               (pImg.y() - _patchSizeHalf + i + 1) *_scale,
                                                                               (pImg.x() - _patchSizeHalf + j) *_scale);
                                const double fy0 = math::bilinearInterpolation(mat,
                                                                               (pImg.y() - _patchSizeHalf + i ) *_scale,
                                                                               (pImg.x() - _patchSizeHalf + j ) *_scale);

                                double dx = 0.5 * (fx1 - fx0);
                                double dy = 0.5 * (fy1 - fy0);
                                _jacobian.col(idxObservation*_patchArea + idxPixel) = (dx*jacobianFeature.row(0) + dy*jacobianFeature.row(1))*(frameRef->camera()->focalLength() / _scale);

                            }
                        }
                        _patchesRef[idxObservation] = patchRef;

                    }else{
                        int idxPixel = 0;
                        for (int i = 0; i < _patchSize; i++)
                        {
                            for (int j = 0; j < _patchSize; j++)
                            {
                                _jacobian.col(idxObservation*_patchArea + idxPixel) = 0.0 * Eigen::Matrix<double,6,1>::Ones();

                            }
                        }
                    }
                }
            }


        }

        bool Evaluate(double const* const* parameters,
                        double* residuals,
                        double** jacobians) const {

            for ( int idxF = 0; idxF < _featuresWithPoints.size(); idxF++)
            {
                Sophus::SE3d pose;
                const auto pCamera = pose * _featuresWithPoints[idxF]->point()->position();
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
                            double grayTarget =     math::bilinearInterpolation(_frameRef->grayImage(_level),
                                                                                (pImg.y() - _patchSizeHalf + i) *_scale,
                                                                                (pImg.x() - _patchSizeHalf + j) *_scale);
                            residuals[idxF*_patchArea + idxPixel] = (_patchesRef[idxF](i,j) - grayTarget);
                        }
                    }
                }else{
                    int idxPixel = 0;
                    for (int i = 0; i < _patchSize; i++)
                    {
                        for (int j = 0; j < _patchSize; j++)
                        {
                            residuals[idxF*_patchArea + idxPixel] = 0.0;

                        }
                    }
                }

            }
            return true;

        }

    };



    Sophus::SE3d ImageAlignment::align(Frame::ShConstPtr referenceFrame, Frame::ShConstPtr targetFrame) {

        for (uint32_t level=_levelMax; level >= _levelMin; level--)
        {

            double pose[6];
            ceres::Problem problem;
            problem.AddResidualBlock(new PhotometricLoss(referenceFrame,targetFrame,_patchSize,level),new ceres::HuberLoss(10.0),&pose[0]);
            ceres::Solver::Options options;
            options.linear_solver_type = ceres::DENSE_SCHUR;
            options.minimizer_progress_to_stdout = true;
            ceres::Solver::Summary summary;
            ceres::Solve(options, &problem, &summary);
        }
        return Sophus::SE3d();
    }

    ImageAlignment::ImageAlignment(uint32_t levelMax, uint32_t levelMin, uint32_t patchSize)
    : _levelMax(levelMax),
    _levelMin(levelMin),
    _patchSize(patchSize){

    }
}
}
