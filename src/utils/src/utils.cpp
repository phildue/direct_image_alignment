//
// Created by phil on 08.08.21.
//

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <Eigen/Dense>
#include <opencv4/opencv2/core/eigen.hpp>
#include "utils/utils.h"
#include "Exceptions.h"
#include <filesystem>
namespace fs = std::filesystem;
namespace pd{ namespace vision {
        

        void utils::throw_if_nan(const Eigen::MatrixXd &mat, const std::string &msg)
        {
            auto result = mat.norm();
            if (std::isnan(mat.norm()) || std::isinf(result))
            {
                std::stringstream ss;
                ss << mat;
                throw pd::Exception(msg + " contains nan: \n" + ss.str());
            }
        }

        Image utils::loadImage(const fs::path& path, int height, int width, bool grayscale)
        {
            auto mat = cv::imread(path.string(), grayscale ? cv::IMREAD_GRAYSCALE : cv::IMREAD_COLOR);

            if (mat.cols == 0 || mat.rows == 0)
            {
                throw pd::Exception("Failure during load of: " + path.string());
            }

            if ( height > 0 && width > 0)
            {
                cv::resize(mat,mat,cv::Size(width,height));
            }

            Image img;
            cv::cv2eigen(mat,img);
            return img;
        }

        Eigen::MatrixXd utils::loadDepth(const fs::path& path, int height, int width )
        {
            auto mat = cv::imread(path.string(), cv::IMREAD_ANYDEPTH);

            if (mat.cols == 0 || mat.rows == 0)
            {
                throw pd::Exception("Failure during load of: " + path.string());
            }

            if ( height > 0 && width > 0)
            {
                cv::resize(mat,mat,cv::Size(width,height));
            }

            Eigen::MatrixXd img;
            cv::cv2eigen(mat,img);
            return img;

        }

        void utils::saveImage(const Image& img,const fs::path& path)
        {
            cv::Mat mat;
            cv::eigen2cv(img,mat);
            mat.convertTo(mat,CV_8UC3);
            cv::imwrite(path.string() + ".png",mat);
        }
        void utils::saveDepth(const Eigen::MatrixXd& img,const fs::path& path)
        {
            cv::Mat mat;
            cv::eigen2cv(img,mat);
            mat.convertTo(mat,CV_32F);
            cv::imwrite(path.string() + ".exr",mat);
        }


    }}