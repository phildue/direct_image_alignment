cmake_minimum_required(VERSION 3.8)
project(lukas_kanade VERSION 1.0.0)

set(sources
        src/Warp.cpp
        ${CMAKE_BINARY_DIR}/include/easylogging++.cc
        )

set(headers
        src/LukasKanade.hpp
        src/LukasKanade.h
        src/LukasKanadeInverseCompositional.hpp
        src/LukasKanadeInverseCompositional.h
        src/Warp.h
        include/lukas_kanade/lukas_kanade.h
        )

pd_setup_lib(${PROJECT_NAME} ${PROJECT_VERSION} "${sources}" "${headers}" pd)
target_link_libraries(${PROJECT_NAME} core utils solver)
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

        pd_add_test(lukas_kanade_flow lukas_kanade)
        pd_add_test(lukas_kanade_affine lukas_kanade)
        pd_add_test(lukas_kanade_se3 lukas_kanade)
        
        target_compile_definitions(lukas_kanade_affineTest PUBLIC TEST_RESOURCE="${CMAKE_CURRENT_LIST_DIR}/test/resource")
        target_compile_definitions(lukas_kanade_flowTest PUBLIC TEST_RESOURCE="${CMAKE_CURRENT_LIST_DIR}/test/resource")
        target_compile_definitions(lukas_kanade_se3Test PUBLIC TEST_RESOURCE="${CMAKE_CURRENT_LIST_DIR}/test/resource")
endif()
