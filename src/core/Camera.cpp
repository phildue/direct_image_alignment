//
// Created by phil on 30.06.21.
//

#include "Camera.h"
namespace pd{
    namespace vision {

        Eigen::Vector2d Camera::camera2image(const Eigen::Vector3d &pWorld) const {
            if (pWorld.z() <= 0)
            {
                return {std::numeric_limits<double>::quiet_NaN(),std::numeric_limits<double>::signaling_NaN()};
            }
            Eigen::Vector3d pProj = _K * pWorld;
            return {pProj.x()/pProj.z(), pProj.y()/pProj.z()};
        }

        Eigen::Vector3d Camera::image2camera(const Eigen::Vector2d &pImage, double depth) const {
            return image2ray(pImage) * depth;
        }
        Eigen::Vector3d Camera::image2ray(const Eigen::Vector2d &pImage) const {
            return _Kinv * Eigen::Vector3d({pImage.x(),pImage.y(),1});
        }

        Eigen::Matrix<double, 2, 6> Camera::J_xyz2uv(const Eigen::Vector3d &pCamera, double scale) const {
            Eigen::Matrix<double, 2, 6> jacobian;
            const double& x = pCamera.x();
            const double& y = pCamera.y();
            const double z_inv = 1./pCamera.z();
            const double z_inv_2 = z_inv*z_inv;

            jacobian(0,0) = -z_inv;              // -1/z
            jacobian(0,1) = 0.0;                 // 0
            jacobian(0,2) = x*z_inv_2;           // x/z^2
            jacobian(0,3) = y*jacobian(0,2);            // x*y/z^2
            jacobian(0,4) = -(1.0 + x*jacobian(0,2));   // -(1.0 + x^2/z^2)
            jacobian(0,5) = y*z_inv;             // y/z

            jacobian(1,0) = 0.0;                 // 0
            jacobian(1,1) = -z_inv;              // -1/z
            jacobian(1,2) = y*z_inv_2;           // y/z^2
            jacobian(1,3) = 1.0 + y*jacobian(1,2);      // 1.0 + y^2/z^2
            jacobian(1,4) = -jacobian(0,3);             // -x*y/z^2
            jacobian(1,5) = -x*z_inv;            // x/z
            return jacobian * focalLength() / scale;
        }

        Camera::Camera(double f, double cx, double cy) {
            _K << f, 0, cx,
               0, f, cy,
               0, 0, 1;
            _Kinv = _K.inverse();
        }


    }}