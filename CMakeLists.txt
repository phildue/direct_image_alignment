cmake_minimum_required(VERSION 3.8)
project(vslam VERSION 1.0.0)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/tools
		${CMAKE_BINARY_DIR}/)
include(Macros)

set(sources
		src/core/Frame.cpp
		src/core/Feature2D.cpp
		src/core/Point3D.cpp
        src/core/Camera.cpp
		src/core/algorithm.cpp
		src/utils/Log.cpp
		src/system/StereoAlignment.cpp
		src/image_alignment/ImageAlignment.cpp
		src/feature_extraction/FeatureExtraction.cpp
		)

set(headers
		src/core/Frame.h
		src/core/Feature2D.h
		src/core/Point3D.h
        src/core/Camera.h
		src/core/algorithm.h
		src/utils/Exceptions.h
		src/utils/Log.h
		src/image_alignment/ImageAlignment.h
		src/feature_extraction/FeatureExtraction.h
		src/system/StereoAlignment.h
		include/vslam/vslam.h
		)

pd_setup_lib(${PROJECT_NAME} ${PROJECT_VERSION} "${sources}" "${headers}" pd)


# Dependencies


include(conanbuildinfo)
conan_basic_setup(TARGETS)

#Cannot get ceres via conan since package is broken
find_package(Ceres REQUIRED)
target_link_libraries(${PROJECT_NAME} Ceres::ceres)
#target_link_libraries(${PROJECT_NAME} CONAN_PKG::ceres-solver)

if (TARGET Eigen3::Eigen)
	target_link_libraries(${PROJECT_NAME} Eigen3::Eigen)
else()
	target_link_libraries(${PROJECT_NAME} CONAN_PKG::Eigen3)
endif()

target_link_libraries(${PROJECT_NAME} CONAN_PKG::sophus)
target_link_libraries(${PROJECT_NAME} CONAN_PKG::glog)


# Add tests
option(MyLibrary_BUILD_TESTS "Enable to build tests" ON)
if(MyLibrary_BUILD_TESTS)
	enable_testing()
	add_subdirectory(test)
endif()
