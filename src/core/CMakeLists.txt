cmake_minimum_required(VERSION 3.8)
project(core VERSION 1.0.0)

set(sources
        src/Camera.cpp
        src/algorithm.cpp
        )

set(headers
        src/Camera.h
        src/algorithm.h
        src/Exceptions.h
        include/core/core.h
        )
pd_setup_lib(${PROJECT_NAME} ${PROJECT_VERSION} "${sources}" "${headers}" pd)

if (UNIX)
	target_link_libraries(${PROJECT_NAME} stdc++fs)
endif()

find_package(Eigen3 3.4 REQUIRED)
target_link_libraries(${PROJECT_NAME} Eigen3::Eigen3)

find_package(Sophus REQUIRED)
target_link_libraries(${PROJECT_NAME} Sophus::Sophus)

# Add tests
option(VSLAM_BUILD_TESTS "Enable to build tests" ON)
if(VSLAM_BUILD_TESTS)
	enable_testing()
        find_package(GTest CONFIG REQUIRED)

        pd_add_test(algorithm core)
        pd_add_test(camera core)
        pd_add_test(random core)
        
endif()
