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
#include "core/algorithm.h"

INITIALIZE_EASYLOGGINGPP


namespace pd{ namespace vision {
std::map<std::string, std::shared_ptr<Log>> Log::_logs = {};
std::map<std::string, std::shared_ptr<LogCsv>> Log::_logsCsv = {};
std::map<std::string, std::shared_ptr<LogImage>> Log::_logsImage = {};

       
    std::shared_ptr<Log> Log::get(const std::string& name)
    {
        auto it = _logs.find(name);
        if (it != _logs.end())
        {
            return it->second;
        }else{
            auto log = std::make_shared<Log>(name);
            _logs[name] = log;
            return log;
        }
        
    }

    std::shared_ptr<LogImage> Log::getImageLog(const std::string& name, Level block, Level show)
    {
        auto it = _logsImage.find(name);
        if (it != _logsImage.end())
        {
            return it->second;
        }else{
            //TODO check which label is configured
            auto log = std::make_shared<LogImage>(name);
            _logsImage[name] = log;
            return log;
        }
    }
    std::shared_ptr<LogCsv> Log::getCsvLog(const std::string& name, const std::vector<std::string>& header, Level level)
    {
        auto it = _logsCsv.find(name);
        if (it != _logsCsv.end())
        {
            return it->second;
        }else{
            //TODO check which label is configured

            auto log = std::make_shared<LogCsv>(name,header);
            _logsCsv[name] = log;
            return log;
        }
    }


    Log::Log(const std::string& name)
    : _name(name)
    {
        el::Configurations config(CFG_DIR"/log/" + name + ".conf");
        // Actually reconfigure all loggers instead
        el::Loggers::reconfigureLogger(name, config);
      
    }

   




    LogCsv::LogCsv(const std::string& fileName, const std::vector<std::string>& header, const std::string& delimiter)
    : _nElements(header.size())
    , _delimiter(delimiter)
    {
        
        std::stringstream ss;
        for (const auto& e: header)
        {
            ss << e << _delimiter;
        }
        _file.open(fileName,std::ios::out);
        _file << ss.str() << std::endl;
        _file.close();
    }

    void LogCsv::append(const std::vector<std::string>& elements)
    {
        if ( elements.size() != _nElements)
        {
            throw pd::Exception("Number of elements does not match header size: " + std::to_string(elements.size())+ "/" + std::to_string(_nElements));
        }
        std::stringstream ss;
        for (const auto& e: elements)
        {
            ss << e << _delimiter;
        }
        append(ss);
    }

    void LogCsv::append(const std::vector<double>& elements)
    {
        if ( elements.size() != _nElements)
        {
            throw pd::Exception("Number of elements does not match header size: " + std::to_string(elements.size())+ "/" + std::to_string(_nElements));
        }
        std::stringstream ss;
        for (const auto& e: elements)
        {
            ss << e << _delimiter;
        }
        append(ss);
    }

    void LogCsv::append(const std::stringstream& ss)
    {
        _file.open(_fileName,std::ios::app);
        _file << ss.str() << std::endl;
        _file.close();
    }

    LogImage::LogImage(const std::string& name)
    : _folder(LOG_DIR "/" + name)
    , _name(name)
    , _blockLevel(el::Level::Debug)
    , _blockLevelDes(el::Level::Info)
    , _showLevel(el::Level::Info)
    , _showLevelDes(el::Level::Info)
    {}


    void LogImage::logMat(const cv::Mat &mat)
    {
        if (mat.cols == 0 || mat.rows == 0)
        {
            throw pd::Exception("Image is empty!");
        }
        if (_showLevel <= _showLevelDes) {
            cv::imshow(_name, mat);
           // cv::waitKey(_blockLevel <= _blockLevelDes ? -1 : 30);
            cv::waitKey(30);
        }
    }
}}

