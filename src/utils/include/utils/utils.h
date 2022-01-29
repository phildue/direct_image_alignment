#ifndef VSLAM_UTILS_H__
#define VSLAM_UTILS_H__

#include "Log.h"
#include "visuals.h"
#include "Exceptions.h"

#include <string>
#include <filesystem>
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#include <Eigen/Dense>

#include "Exceptions.h"
#include "Log.h"
namespace pd{ namespace vision { namespace utils{
    void throw_if_nan(const Eigen::MatrixXd &mat, const std::string &msg);
    Image loadImage(const fs::path& path, int height = -1, int width = -1, bool grayscale = true);
    Eigen::MatrixXd loadDepth(const fs::path& path, int height = -1, int width = -1);

    void saveImage(const Image& img,const fs::path& path);
    void saveDepth(const Eigen::MatrixXd& img,const fs::path& path);
}}}
#endif