#ifndef VSLAM_UTILS_H__
#define VSLAM_UTILS_H__

#include <string>
#include <filesystem>
#include <map>

#include <Eigen/Dense>

#include "Log.h"
#include "visuals.h"
#include "Exceptions.h"

namespace pd::vslam::utils{
    void throw_if_nan(const Eigen::MatrixXd &mat, const std::string &msg);
    Image loadImage(const std::filesystem::path& path, int height = -1, int width = -1, bool grayscale = true);
    Eigen::MatrixXd loadDepth(const std::filesystem::path& path, int height = -1, int width = -1);
    std::map<Timestamp,SE3d> loadTrajectory(const std::filesystem::path& path);

    void saveImage(const Image& img,const std::filesystem::path& path);
    void saveDepth(const Eigen::MatrixXd& img,const std::filesystem::path& path);
}
#endif