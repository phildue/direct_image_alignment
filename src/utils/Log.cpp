//
// Created by phil on 07.08.21.
//
#include <Eigen/Dense>
#include <opencv4/opencv2/core/eigen.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "Log.h"
#include "core/Frame.h"
#include "Exceptions.h"
#include "core/Point3D.h"
#include "easylogging++.h"

INITIALIZE_EASYLOGGINGPP
namespace pd{ namespace vision {

        int Log::_showLevel = 3;
        int Log::_blockLevel = 3;

        void Log::init(int loglevel, int showLevel, int blockLevel) {
            // Load configuration from file
            el::Configurations conf(CFG_DIR"/log.conf");
            // Reconfigure single logger
            el::Loggers::reconfigureLogger("default", conf);
            // Actually reconfigure all loggers instead
            el::Loggers::reconfigureAllLoggers(conf);
            // Now all the loggers will use configuration from file
            _showLevel = showLevel;
            _blockLevel = blockLevel;
        }


        void Log::logFeatures(std::shared_ptr<const Frame> frame, int radius, int level,bool gradient, const std::string &name) {
            if (level <= FLAGS_v)
            {
                cv::Mat mat;
                if ( gradient )
                {
                    cv::eigen2cv(frame->gradientImage(),mat);
                }else{
                    cv::eigen2cv(frame->grayImage(),mat);
                }

                if (mat.cols == 0 || mat.rows == 0)
                {
                    throw pd::Exception("Image is empty!");
                }

                for (const auto& ft : frame->features())
                {
                    cv::rectangle(mat,cv::Point(ft->position().x() - radius,ft->position().y() - radius),
                            cv::Point(ft->position().x() + radius,ft->position().y() + radius),cv::Scalar(255,255,255));
                }

                logMat(mat, level, name);

            }
        }

        void Log::logFeaturesWithPoints(std::shared_ptr<const Frame> frame, int radius, int level, const std::string &name) {
            if (level <= FLAGS_v)
            {
                cv::Mat mat;
                cv::eigen2cv(frame->gradientImage(),mat);

                if (mat.cols == 0 || mat.rows == 0)
                {
                    throw pd::Exception("Image is empty!");
                }

                for (const auto& ft : frame->features())
                {
                    if (ft->point())
                    {
                        cv::circle(mat,cv::Point(ft->position().x(),ft->position().y()),
                                      radius,cv::Scalar(255,255,255));
                    }else{
                        cv::rectangle(mat,cv::Point(ft->position().x() - radius,ft->position().y() - radius),
                                      cv::Point(ft->position().x() + radius,ft->position().y() + radius),cv::Scalar(255,255,255));
                    }

                }
                logMat(mat, level, name);

            }
        }

        void Log::logFrame(std::shared_ptr<const Frame> frame, int level, const std::string &name) {
            if (level <= FLAGS_v)
            {
                cv::Mat mat;
                cv::eigen2cv(frame->grayImage(),mat);
                logMat(mat, level, name);
            }

        }

        void
        Log::logReprojection(std::shared_ptr<const pd::vision::Frame> frame0,
                             std::shared_ptr<const pd::vision::Frame> frame1, int radius, int level,
                             const std::string &name)  {
            if (level <= FLAGS_v)
            {
                cv::Mat mat;
                cv::eigen2cv(frame1->grayImage(),mat);

                if (mat.cols == 0 || mat.rows == 0)
                {
                    throw pd::Exception("Image is empty!");
                }

                for (const auto& ft : frame0->features())
                {
                    if (ft->point())
                    {
                        auto reprojectedPosition = frame1->world2image(ft->point()->position());
                        cv::rectangle(mat,
                                cv::Point(reprojectedPosition.x()-radius,reprojectedPosition.y()-radius),
                                      cv::Point(reprojectedPosition.x()+radius,reprojectedPosition.y()+radius),cv::Scalar(255,255,255));
                    }

                }

                logMat(mat, level, name);
            }
        }

        void Log::logMat(const Image& matEigen, int level, const std::string &name )
        {
            if (level <= FLAGS_v)
            {
                cv::Mat mat;
                cv::eigen2cv(matEigen,mat);

                logMat(mat, level, name);
            }
        }


        void Log::logMat(const cv::Mat &mat, int level, const std::string &name)
        {
            if (mat.cols == 0 || mat.rows == 0)
            {
                throw pd::Exception("Image is empty!");
            }
            if (level <= _showLevel) {
                cv::imshow(name, mat);
                cv::waitKey(level <= _blockLevel ? -1 : 10);
            }
        }

    }}

