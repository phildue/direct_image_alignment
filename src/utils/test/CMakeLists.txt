cmake_minimum_required(VERSION 3.8)
project(lukas_kanade VERSION 1.0.0)

set(sources
        src/lukas_kanade/Warp.cpp
        ${CMAKE_BINARY_DIR}/include/easylogging++.cc
        )

set(headers
        src/lukas_kanade/LukasKanade.hpp
        src/lukas_kanade/LukasKanade.h
        src/lukas_kanade/LukasKanadeInverseCompositional.hpp
        src/lukas_kanade/LukasKanadeInverseCompositional.h
        src/lukas_kanade/Warp.h
        include/lukas_kanade/lukas_kanade.h
        )

pd_setup_lib(${PROJECT_NAME} ${PROJECT_VERSION} "${sources}" "${headers}" pd)

if (UNIX)
	target_link_libraries(${PROJECT_NAME} stdc++fs)
endif()

#Cannot get ceres via conan since package is broken
#find_package(Ceres REQUIRED)
#target_link_libraries(${PROJECT_NAME} Ceres::ceres)
#target_link_libraries(${PROJECT_NAME} CONAN_PKG::ceres-solver)


# Try for OpenCV 4.X, but settle for whatever is installed
find_package(OpenCV 4 QUIET)
if (NOT OpenCV_FOUND)
  find_package(OpenCV REQUIRED)
endif ()
message(STATUS "Found OpenCV version ${OpenCV_VERSION}")
target_include_directories(${PROJECT_NAME} PUBLIC ${OpenCV_INCLUDE_DIRS})

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBRARIES})
find_package(Eigen3 3.4 REQUIRED)
target_link_libraries(${PROJECT_NAME} Eigen3::Eigen3)

find_package(Sophus REQUIRED)
find_package(easyloggingpp REQUIRED)
target_link_libraries(${PROJECT_NAME} Sophus::Sophus)
target_link_libraries(${PROJECT_NAME} easyloggingpp::easyloggingpp)

find_package(matplotlib_cpp)
target_link_libraries(${PROJECT_NAME} matplotlib_cpp::matplotlib_cpp)

# Add tests
option(VSLAM_BUILD_TESTS "Enable to build tests" ON)
if(VSLAM_BUILD_TESTS)
	enable_testing()
	add_subdirectory(test)
endif()
