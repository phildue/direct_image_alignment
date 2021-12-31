//
// Created by phil on 07.08.21.
//

#ifndef VSLAM_LOG_H
#define VSLAM_LOG_H

#include <memory>
#include <vector>
#include <string>
#include <map>
#include <iostream>
#include "matplotlibcpp.h"

#include <opencv2/opencv.hpp>
#include "core/types.h"

#include "easylogging++.h"

#define SYSTEM(loglevel) CLOG(loglevel, "system")
#define IMAGE_ALIGNMENT(loglevel) CLOG(loglevel, "image_alignment")
#define SOLVER(loglevel) CLOG(loglevel, "solver")


namespace pd{ namespace vision{
    class Frame;
    class FrameRGBD;
    using Level = el::Level;
    namespace plt = matplotlibcpp;

    class LogCsv
    {
        public:
        LogCsv(const std::string& file, const std::string& delimiter = ";");
        void setHeader(const std::vector<std::string>& header);
        virtual void append(const std::vector<std::string>& elements);
        virtual void append(const std::vector<double>& elements);
        private:
        const std::string _name;
        const std::string _fileName;
        
        std::fstream _file;

        int _nElements;
        const std::string _delimiter;
        void append(const std::stringstream& ss);
    };

    class LogCsvNull : public LogCsv
    {
        void append(const std::vector<std::string>& elements) override {}
    };

    template<typename... Args> 
    using DrawFunctor = cv::Mat(*)(Args... args);

    class LogImage
    {
        public:
        LogImage(const std::string& name, bool block = false, bool show = true, bool save = false);
        template<typename... Args>
        void append(DrawFunctor<Args...> draw,Args... args)
        {
            if (_show)
            {
                logMat(draw(args...));
            }
        }
        void append(const cv::Mat& mat)
        {
            if (_show || _save)
            {
                logMat(mat);
            }         
        }
        bool _block;
        bool _show;
        bool _save;
        protected:
        const std::string _name;
        const std::string _folder;
        std::uint32_t _ctr;

        void logMat(const cv::Mat &mat);

    };
  
    class Log {
    public:
    static std::shared_ptr<Log> get(const std::string& name,Level level = el::Level::Info);
    static std::shared_ptr<LogImage> getImageLog(const std::string& name, Level level = el::Level::Info);
    static std::shared_ptr<LogCsv> getCsvLog(const std::string& name, Level level);
    
    Log(const std::string& name);

     private:

    const std::string _name;
    static std::map<std::string, std::map<Level,std::shared_ptr<Log>>> _logs;
    static std::map<std::string, std::map<Level,std::shared_ptr<LogCsv>>> _logsCsv;
    static std::map<std::string, std::map<Level,std::shared_ptr<LogImage>>> _logsImage;

    };
    }}

#endif //VSLAM_LOG_H
