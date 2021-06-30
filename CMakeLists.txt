cmake_minimum_required(VERSION 3.8)
project(direct_image_alignment VERSION 1.0.0)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/tools)
include(Macros)

set(sources
		src/Frame.cpp
		src/ImageAlignment.cpp
		src/Feature2D.cpp
		src/Point3D.cpp
		src/Pose.cpp
		src/Camera.cpp
		)

set(headers
		src/Frame.h
		src/ImageAlignment.h
		src/Feature2D.h
		src/Point3D.h
		src/Pose.h
		src/Camera.h
		include/direct_image_alignment/direct_image_alignment.h
		)

setup_package(${PROJECT_NAME} ${PROJECT_VERSION} "${sources}" "${headers}" PD)


# Dependencies
find_package(Eigen3 3.3 REQUIRED)
target_link_libraries(${PROJECT_NAME} Eigen3::Eigen)

# Add tests
option(MyLibrary_BUILD_TESTS "Enable to build tests" ON)
if(MyLibrary_BUILD_TESTS)
	enable_testing()
	add_subdirectory(test)
endif()
