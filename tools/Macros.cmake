function(pd_setup_lib name version sources headers namespace)

# Construct library from sources
add_library(${name}
    ${sources}
    ${headers}
    )
set_property(TARGET ${name} PROPERTY POSITION_INDEPENDENT_CODE ON)
# Configure alias so there is no difference whether we link from source/from already built
add_library(${namespace}::${name} ALIAS ${name})

# Set include path (can be different in build/install)
target_include_directories(${name} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/src/>
    $<INSTALL_INTERFACE:include/${name}>
    )
target_include_directories(${name} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include/>
    $<INSTALL_INTERFACE:include>
    )
# Install everything a user of the library needs
install(TARGETS ${name} EXPORT ${name}Targets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    )

install(EXPORT ${name}Targets
    DESTINATION lib/cmake/${name}
    FILE ${name}Targets.cmake
    NAMESPACE ${namespace}::
    DESTINATION share/${name}/cmake
    )

install(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/src/
    DESTINATION include/${name}
    FILES_MATCHING # install only matched files
    PATTERN "*.h" # select header files
    )
install(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/src/
    DESTINATION include/${name}
    FILES_MATCHING # install only matched files
    PATTERN "*.hpp" # select header files
    )  
install(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/include/
    DESTINATION include/
    FILES_MATCHING # install only matched files
    PATTERN "*.h" # select header files
    )

install(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/cfg/
    DESTINATION share/${name}/cfg/
    FILES_MATCHING # install only matched files
    PATTERN "*" # select all files
    )

include(CMakePackageConfigHelpers)

write_basic_package_version_file(${name}ConfigVersion.cmake
    VERSION ${version}
    COMPATIBILITY SameMajorVersion
    )

# install export target and config for find_package
include(CMakePackageConfigHelpers)
configure_package_config_file(
	"${CMAKE_CURRENT_LIST_DIR}/tools/${name}Config.cmake.in" "${CMAKE_CURRENT_BINARY_DIR}/${name}Config.cmake"
	INSTALL_DESTINATION "share/${name}/cmake/"
)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/${name}Config.cmake" DESTINATION "share/${name}/cmake/")


endfunction()


macro(pd_add_test unit lib)
	add_executable(${unit}Test
			test_${unit}.cpp
			)

	target_link_libraries(${unit}Test
			PRIVATE
			pd::${lib}
			GTest::gtest_main
			)

	add_test(NAME ${unit}.UnitTest
			COMMAND ${unit}Test
			)

endmacro()

