//
// Created by phil on 07.08.21.
//
#include <eigen3/Eigen/Dense>
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
std::map<std::string, std::map<Level,std::shared_ptr<Log>>> Log::_logs = {};
std::map<std::string, std::map<Level,std::shared_ptr<LogCsv>>> Log::_logsCsv = {};
std::map<std::string, std::map<Level,std::shared_ptr<LogImage>>> Log::_logsImage = {};
Level Log::_blockLevel = Level::Unknown;
Level Log::_showLevel = Level::Debug;
       
    std::shared_ptr<Log> Log::get(const std::string& name,Level level)
    {
        auto it = _logs.find(name);
        if (it != _logs.end())
        {
            return it->second[level];
        }else{
            std::map<Level,std::shared_ptr<Log>> log = {
                {el::Level::Debug,std::make_shared<Log>(name)},
                {el::Level::Info,std::make_shared<Log>(name)},
                {el::Level::Warning,std::make_shared<Log>(name)},
                {el::Level::Error,std::make_shared<Log>(name)},

            };
            _logs[name] = log;
            return log[level];
        }
        
    }

    std::shared_ptr<LogImage> Log::getImageLog(const std::string& name, Level level)
    {
        auto it = _logsImage.find(name);
        if (it != _logsImage.end())
        {
            return it->second[level];
        }else{
            const std::vector<Level> levels = {
                Level::Debug,
                Level::Info,
                Level::Warning,
                Level::Error
            };
            std::map<Level,std::shared_ptr<LogImage>> log;
            for (const auto & l : levels)
            {
                log[l] = std::make_shared<LogImage>(name,l >= _blockLevel, l >= _showLevel);
            }
             
            _logsImage[name] = log;
            return log[level];
        }
    }
    std::shared_ptr<LogCsv> Log::getCsvLog(const std::string& name, Level level)
    {
        auto it = _logsCsv.find(name);
        if (it != _logsCsv.end())
        {
            return it->second[level];
        }else{
            std::map<Level,std::shared_ptr<LogCsv>> log = {
                {el::Level::Debug,std::make_shared<LogCsv>(name)},
                {el::Level::Info,std::make_shared<LogCsv>(name)},
                {el::Level::Warning,std::make_shared<LogCsv>(name)},
                {el::Level::Error,std::make_shared<LogCsv>(name)},

            };
            _logsCsv[name] = log;
            return log[level];
        }
    }


    Log::Log(const std::string& name)
    : _name(name)
    {
        el::Configurations config(CFG_DIR"/log/" + name + ".conf");
        // Actually reconfigure all loggers instead
        el::Loggers::reconfigureLogger(name, config);
      
    }

    LogCsv::LogCsv(const std::string& fileName, const std::string& delimiter)
    : _nElements(0)
    , _delimiter(delimiter)
    , _fileName(fileName)
    {
       
    }
    void LogCsv::setHeader(const std::vector<std::string>& header)
    {
        _nElements = header.size();
        std::stringstream ss;
        for (const auto& e: header)
        {
            ss << e << _delimiter;
        }
        _file.open(_fileName,std::ios::out);
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

    LogImage::LogImage(const std::string& name,bool block, bool show, bool save)
    : _block(block)
    , _show(show)
    , _save(save)
    , _name(name)
    , _folder(LOG_DIR "/" + name)
    , _ctr(0U)
    {}


    void LogImage::logMat(const cv::Mat &mat)
    {
        if (mat.cols == 0 || mat.rows == 0)
        {
            throw pd::Exception("Image is empty!");
        }
        if (_show)
        {
            cv::imshow(_name, mat);
            // cv::waitKey(_blockLevel <= _blockLevelDes ? -1 : 30);
            cv::waitKey(_block ? 0 : 30);
        }
        if (_save)
        {
            cv::imwrite(_folder + "/" + _name + std::to_string(_ctr++) +".jpg",mat);
        }
       
    }
}}

