#include <utility>
#include <iostream>

#include <ceres/ceres.h>
#include <ceres/cubic_interpolation.h>
#include <ceres/rotation.h>

#include "core/Camera.h"
#include "core/algorithm.h"
#include "core/Point3D.h"
#include "core/Feature2D.h"
#include "core/Frame.h"

#include "utils/utils.h"

#include "solver/LeastSquaresSolver.h"
#include "ImageAlignmentDense.h"

namespace pd{ namespace vision{



class OptimizerHandle
{
    const int _stepSizeX, _stepSizeY;
    const std::uint32_t _level;
    const double _scale;
    Eigen::MatrixXd _jacobian;
    Eigen::Matrix<double, 3, Eigen::Dynamic> _pointsCamRef;
    Eigen::Matrix<int, 2, Eigen::Dynamic> _pointsImageRef;
    Eigen::VectorXd _valid;
    const Frame::ShConstPtr _frameTarget;
    const FrameRGBD::ShConstPtr _frameRef;

public:
    OptimizerHandle(FrameRGBD::ShConstPtr referenceFrame, Frame::ShConstPtr targetFrame, int level)
    :_stepSizeX(1)
    ,_stepSizeY(1)
    , _level(level)
    , _scale( 1.0/(1U<<level))
    , _frameRef(referenceFrame)
    , _frameTarget(targetFrame)
    {


        Sophus::SE3d T = algorithm::computeRelativeTransform(referenceFrame->pose(), targetFrame->pose());
        const int nPixels = (referenceFrame->width(_level)-2)/_stepSizeX*(referenceFrame->height(level)-2)/_stepSizeY;

        _jacobian.conservativeResize(nPixels,Sophus::SE3d::DoF);
        _pointsCamRef.conservativeResize(Eigen::NoChange,nPixels);
        _valid.conservativeResize(nPixels);
        _pointsImageRef.conservativeResize(Eigen::NoChange,nPixels);

        const auto& mat = referenceFrame->grayImage(_level);
        int idxPixel = 0;
        for (int v = 1; v < referenceFrame->height(_level)-1; v += _stepSizeY)
        {
            for (int u = 1; u < referenceFrame->width(_level)-1; u += _stepSizeX)
            {
                const double depth = referenceFrame->depthMap(_level)(v,u);

                if ( std::isnan(depth) || depth <= 0.1 || std::isinf(depth))
                {
                    _valid(idxPixel) = 0;

                }else{

                    _valid(idxPixel) = 1;

                    _pointsImageRef.col(idxPixel) = Eigen::Matrix<int,2,1>(u,v);
                    _pointsCamRef.col(idxPixel) = referenceFrame->image2camera({u,v},depth);


                    utils::throw_if_nan(_pointsCamRef.col(idxPixel), "3D Point");

                    const auto J_xyz2uv = referenceFrame->camera()->J_xyz2uv(_pointsCamRef.col(idxPixel), _scale);

                    VLOG(5) << "scale: " << _scale << "p3d:" << _pointsCamRef.col(idxPixel).transpose() << "J_xyz2uv =\n " << J_xyz2uv;

                    utils::throw_if_nan(J_xyz2uv, "Point Jac.");

                    const double fu0 = mat(v * _scale, (u - 1) * _scale);
                    const double fu1 = mat(v * _scale, (u + 1) * _scale);
                    const double fv1 = mat((v + 1) * _scale, u * _scale);
                    const double fv0 = mat((v - 1) * _scale, u * _scale);

                    const double du = 0.5 * (fu1 - fu0);
                    const double dv = 0.5 * (fv1 - fv0);

                    _jacobian.row(idxPixel) = (du * J_xyz2uv.row(0) + dv * J_xyz2uv.row(1));
                }
                idxPixel++;




            }
        }
        utils::throw_if_nan(_jacobian, "Jac.");

    }

    bool computeResidual(const Eigen::MatrixXd& x, Eigen::MatrixXd& residual, Eigen::MatrixXd& weights) const
    {

        const Sophus::SE3d pose = Sophus::SE3d::exp(x);

       //TODO const Eigen::Matrix<double,3,Eigen::Dynamic> pointsCamTarget = pose * _pointsCamRef;

        for (int idxPoint = 0; idxPoint < _pointsCamRef.cols(); idxPoint++)
        {
            if (_valid(idxPoint) > 0)
            {
                const Eigen::Vector3d pointsCamTarget = pose * _pointsCamRef.col(idxPoint);
                const Eigen::Vector2d pImgTarget = _frameTarget->camera2image(pointsCamTarget);
                VLOG(5) << "Reprojection: " << pImgTarget.transpose();

                if ( _frameTarget->isVisible(pImgTarget,1,_level))
                {
                    const Eigen::Vector2i& pImgRef = _pointsImageRef.col(idxPoint);
                    const double intensityRef =_frameRef->grayImage(_level)(pImgRef.y(),pImgRef.x());
                    const double intensityTarget = algorithm::bilinearInterpolation(_frameTarget->grayImage(_level),pImgTarget.x(),pImgTarget.y());
                    const double res = intensityRef - intensityTarget;
                    VLOG(5) << "Residual at (" << pImgRef.x() << "," << pImgRef.y() << "): " << intensityRef << "-" << intensityTarget << "=" << res;

                    if (std::isnan(res))
                    {
                        throw pd::Exception("Residual is NaN for: (" + std::to_string( pImgRef.x() ) + "," + std::to_string(pImgRef.y()) + ")");
                    }
                    residual(idxPoint) = res;
                    weights(idxPoint) = 1.0;
                }else
                {
                    weights(idxPoint) = 1.0;
                }
            }else
            {
                weights(idxPoint) = 0;
            }
        }
        return true;
    }

    bool computeJacobian(const Eigen::MatrixXd& x, Eigen::MatrixXd& jacobian)
    {
        jacobian = _jacobian;
        return true;
    }
};

    void ImageAlignmentDense::align(FrameRGBD::ShConstPtr referenceFrame, Frame::ShConstPtr targetFrame) const
    {
        for (int level = _levelMax; level >= _levelMin; --level)
        {
            Sophus::SE3d T = algorithm::computeRelativeTransform(referenceFrame->pose(), targetFrame->pose());
            Eigen::MatrixXd posev6d = T.log();
            VLOG(4) << "IA init: " << " Level: " << level;

            const int nPixels = (referenceFrame->width(level)-2)*(referenceFrame->height(level)-2);

            auto cost = std::make_shared<OptimizerHandle>(referenceFrame,targetFrame,level);
            auto lls = std::make_shared<LeastSquaresSolver>(
                    [&](const Eigen::MatrixXd& x, Eigen::MatrixXd& residual, Eigen::MatrixXd& weights) { return cost->computeResidual(x,residual,weights);},
                    [&](const Eigen::MatrixXd& x, Eigen::MatrixXd& jacobian) { return cost->computeJacobian(x,jacobian);},
                    [&](const Eigen::MatrixXd& dx, Eigen::MatrixXd& x) { x = (Sophus::SE3d::exp(x) * Sophus::SE3d::exp(-dx)).log(); return true;},
                    nPixels,
                    Sophus::SE3d::DoF,
                    0.0001,
                    0.0001,
                    100
            );

            lls->solve(posev6d);

            targetFrame->setPose(Sophus::SE3d::exp(posev6d)*referenceFrame->pose());

            }

    }


        ImageAlignmentDense::ImageAlignmentDense(uint32_t levelMax, uint32_t levelMin)
    : _levelMax(levelMax)
    , _levelMin(levelMin)
    {}

    }
}
