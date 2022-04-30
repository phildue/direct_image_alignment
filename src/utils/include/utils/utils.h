#ifndef VSLAM_UTILS_H__
#define VSLAM_UTILS_H__

#include <string>
#include <filesystem>
#include <map>

#include <Eigen/Dense>

#include <core/core.h>
#include "Log.h"
#include "visuals.h"
#include "Exceptions.h"

namespace pd::vslam::utils{
    void throw_if_nan(const Eigen::MatrixXd &mat, const std::string &msg);
    Image loadImage(const std::filesystem::path& path, int height = -1, int width = -1, bool grayscale = true);
    Eigen::MatrixXd loadDepth(const std::filesystem::path& path, int height = -1, int width = -1);

    /**
     * @brief Load trajectory from file (TUM RGBD Format)
     * 
     * @param path filepath
     * @return std::map<Timestamp,SE3d> 
     */
    std::map<Timestamp,SE3d> loadTrajectory(const std::filesystem::path& path);
    
    /**
     * @brief Write trajectory to txt file (TUM RGBD Format)
     * 
     * @param traj 
     * @param path 
     * @param writeCovariance 
     */
    void writeTrajectory(const Trajectory& traj, const std::filesystem::path& path, bool writeCovariance = false);

    void saveImage(const Image& img,const std::filesystem::path& path);
    void saveDepth(const Eigen::MatrixXd& img,const std::filesystem::path& path);
}
#endif