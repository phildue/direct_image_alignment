#include <utility>

#include <utility>

#include <utility>

#include "ImageAlignment.h"
#include "Camera.h"
#include "math.h"
#include <iostream>
#include <ceres/ceres.h>
#include <ceres/cubic_interpolation.h>
#include <ceres/rotation.h>

namespace pd{
namespace vision{

    struct PhotometricLoss : ceres::CostFunction{
        // gray value of original tag point
        // 3d point in world coordinate
        Eigen::MatrixXd _pointsWorld;
        Eigen::VectorXd _visibility;
        Eigen::MatrixXd _jacobian;
        std::vector<Eigen::MatrixXd> _patchesRef;

        Frame::ShConstPtr  _frameRef,_frameTarget;
        const int _patchSize;
        const int _patchArea;
        const int _patchSizeHalf;
        const int _level;
        const double _scale;
        PhotometricLoss (Frame::ShConstPtr frameRef,
                         Frame::ShConstPtr frameTarget,
                         int patchSize,
                         int level,
                         const ceres::BiCubicInterpolator<ceres::Grid2D<std::uint8_t , 1>>& interpolator):
                        _frameRef(std::move(frameRef)),
                        _frameTarget(std::move(frameTarget)),
                        _patchSize(patchSize),
                        _level(level),
                        _scale(1.0/(1U<<level)),
                        _patchSizeHalf(patchSize/2){

            _patchesRef.resize(frameRef->features().size());
            _jacobian = Eigen::MatrixXd(frameRef->features().size(),patchSize*patchSize*6);
            for ( int idxF = 0; idxF < frameRef->features().size(); idxF++)
            {
                const auto& f = frameRef->features()[idxF];
                if ( f->point() )
                {
                    const auto pCamera = frameRef->pose() * f->point()->position();
                    const auto jacobianFeature = frameRef->camera()->J_xyz2uv(pCamera);
                    Eigen::MatrixXd patchRef(_patchSize,_patchSize);
                    const auto pImg = _frameRef->world2image(f->point()->position());
                    int idxP = 0;
                    for (int i = 0; i < _patchSize; i++)
                    {
                        for (int j = 0; j < _patchSize; j++)
                        {
                            patchRef(i,j) = math::bilinearInterpolation(_frameRef->grayImage(level),
                                                                         (pImg.y() - _patchSizeHalf + i) *_scale,
                                                                         (pImg.x() - _patchSizeHalf + j) *_scale);

                            double dx = 0.5;
                            double dy = 0.5;
                            _jacobian.col(idxF*_patchArea + idxP) = (dx*jacobianFeature.row(0) + dy*jacobianFeature.row(1))*(frameRef->camera()->focalLength() / _scale);

                            //TODO compute Jacobian here
                        }
                    }
                    _patchesRef[idxF] = patchRef;

                    _visibility[idxF] = 1;
                }else{
                    _visibility[idxF] = 0;
                }
            }


        }

        bool Evaluate(double const* const* parameters,
                        double* residuals,
                        double** jacobians) const {

            for ( int idxF = 0; idxF < _frameRef->features().size(); idxF++)
            {
                Sophus::SE3d pose;
                const auto pImg = _frameTarget->camera2image(pose * Eigen::Vector3d(_pointsWorld(idxF)));

                Eigen::MatrixXd _patchTarget(_patchSize,_patchSize);
                for (int i = 0; i < _patchSize; i++)
                {
                    for (int j = 0; j < _patchSize; j++)
                    {
                        auto idxRow = static_cast<std::uint32_t > ( i * _scale);
                        auto idxCol = static_cast<std::uint32_t > ( j * _scale);
                        double grayTarget =     math::bilinearInterpolation(_frameRef->grayImage(_level),
                                                                           (pImg.y() - _patchSizeHalf + i) *_scale,
                                                                           (pImg.x() - _patchSizeHalf + j) *_scale);

                        residuals[i * _patchSize + j] = _patchesRef[idxF](i,j) - grayTarget;
                    }
                }


                return true;
            }
        }

    };

    Pose::ShConstPtr ImageAlignment::align(Frame::ShConstPtr referenceFrame, Frame::ShConstPtr targetFrame) {

        for (int i=_levelMax; i >= _levelMin; i--)
        {

            const auto& img = referenceFrame->grayImage(i);
            // Cubic _interpolator
            // Optimize transform params (6 components):
            // 3 for rotation vector (@sa Rodrigues)
            // 3 for translation vector


            ceres::Problem problem;
            ceres::Solver::Options options;
            options.linear_solver_type = ceres::DENSE_SCHUR;
            options.minimizer_progress_to_stdout = true;
            ceres::Solver::Summary summary;
            ceres::Solve(options, &problem, &summary);
        }
        return pd::vision::Pose::ShConstPtr();
    }
}
}
