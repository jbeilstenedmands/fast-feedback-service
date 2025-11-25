project(import CXX)

set(CMAKE_CXX_STANDARD 20)

# GTest could have been made available under a different name
if(NOT TARGET GTest::gtest_main)
    FetchContent_MakeAvailable(GTest)
endif()
FetchContent_MakeAvailable(Eigen3)

# add_subdirectory(tests)
add_executable(importer
    import.cc
)
target_link_libraries(importer
    PRIVATE
    fmt
    dx2
    h5read
    Eigen3::Eigen
    argparse
    nlohmann_json::nlohmann_json
    spdlog::spdlog
    ffs_common
    ${CMAKE_THREAD_LIBS_INIT}
)