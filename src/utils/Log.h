//
// Created by phil on 07.08.21.
//

#ifndef VSLAM_LOG_H
#define VSLAM_LOG_H

#include <memory>

#include <glog/logging.h>
#include <opencv2/core/mat.hpp>
#include "core/types.h"
namespace pd{ namespace vision{
    class Frame;
    class Log {
    public:
    static void init(int loglevel = 3, int showLevel = 3, int blockLevel = 3);

    static void logFrame(std::shared_ptr<const Frame> frame, int level = 0, const std::string& name = "Log");
    static void logFeatures(std::shared_ptr<const Frame> frame, int radius = 3, int level = 0, bool gradient = true,
                            const std::string &name = "Log");

    static void logFeaturesWithPoints(std::shared_ptr<const Frame> frame, int radius = 3, int level = 0,
                            const std::string &name = "Log");
    static void logReprojection(std::shared_ptr<const Frame> frame0,std::shared_ptr<const Frame> frame1, int radius = 3, int level = 0,
                                      const std::string &name = "Log");

    static void logMat(const Image& mat, int level, const std::string &name );

    static void logMat(const cv::Mat &mat, int level, const std::string &name);
    private:
    static int _showLevel;
    static int _blockLevel;
    };
    }}

#endif //VSLAM_LOG_H
