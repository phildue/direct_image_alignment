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

#include <opencv2/core/mat.hpp>
#include "core/types.h"
#include "easylogging++.h"

#define SYSTEM(loglevel) CLOG(loglevel, "system")
#define IMAGE_ALIGNMENT(loglevel) CLOG(loglevel, "image_alignment")
#define SOLVER(loglevel) CLOG(loglevel, "solver")


namespace pd{ namespace vision{
    class Frame;
    class FrameRGBD;
    using Level = el::Level;

    class LogCsv
    {
        public:
        LogCsv(const std::string& file, const std::vector<std::string>& header,const std::string& delimiter = ";");
        virtual void append(const std::vector<std::string>& elements);
        virtual void append(const std::vector<double>& elements);
        protected:
        const std::string _name;
        const std::string _fileName;
        
        std::fstream _file;

        const int _nElements;
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
        LogImage(const std::string& name);
        template<typename... Args>
        void append(DrawFunctor<Args...> draw,Args... args)
        {
            if (shouldDraw())
            {
                logMat(draw(args...));
            }
        }
        void append(const cv::Mat& mat)
        {
            if (shouldDraw())
            {
                logMat(mat);
            }         
        }

        protected:
        const std::string _name;
        const std::string _folder;
        Level _blockLevel, _showLevel;
        Level _blockLevelDes, _showLevelDes;

        void logMat(const cv::Mat &mat);
        virtual bool shouldDraw() const {return true;}

    };
    class LogImageNull : public LogImage
    {
        protected:
        bool shouldDraw() const final {return false;}
    };
    class Log {
    public:
    static std::shared_ptr<Log> get(const std::string& name);
    static std::shared_ptr<LogImage> getImageLog(const std::string& name, Level show = el::Level::Info, Level block = el::Level::Debug);
    static std::shared_ptr<LogCsv> getCsvLog(const std::string& name,const std::vector<std::string>& header, Level level);

    Log(const std::string& name);

     private:
    int _showLevel;
    int _blockLevel;
    int _logLevel;
    const std::string _name;
    static std::map<std::string, std::shared_ptr<Log>> _logs;
    static std::map<std::string, std::shared_ptr<LogCsv>> _logsCsv;
    static std::map<std::string, std::shared_ptr<LogImage>> _logsImage;

    };
    }}

#endif //VSLAM_LOG_H
