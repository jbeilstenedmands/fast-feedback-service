project(indexer CXX)

set(CMAKE_CXX_STANDARD 20)

find_package (Threads REQUIRED)

# Automatic Dependencies
set(FETCHCONTENT_QUIET OFF)
include(FetchContent)
FetchContent_Declare(
    Eigen3
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0
    EXCLUDE_FROM_ALL
    FIND_PACKAGE_ARGS
)
FetchContent_Declare(
    pocketfft
    GIT_REPOSITORY https://github.com/mreineck/pocketfft
    GIT_TAG cpp
)
FetchContent_Declare(
    mdspan
    GIT_REPOSITORY https://github.com/kokkos/mdspan
    GIT_TAG mdspan-0.6.0
    EXCLUDE_FROM_ALL
    FIND_PACKAGE_ARGS
)
FetchContent_MakeAvailable(mdspan)
FetchContent_MakeAvailable(Eigen3)
FetchContent_MakeAvailable(pocketfft)

FetchContent_GetProperties(pocketfft)
add_library(pocketfft INTERFACE)
target_include_directories(pocketfft INTERFACE ${pocketfft_SOURCE_DIR})

# GTest could have been made available under a different name
if(NOT TARGET GTest::gtest_main)
    FetchContent_MakeAvailable(GTest)
endif()

add_subdirectory(tests)
add_executable(baseline_indexer
    indexer.cc
)
target_link_libraries(baseline_indexer
    PRIVATE
    fmt
    dx2
    Eigen3::Eigen
    pocketfft
    argparse
    nlohmann_json::nlohmann_json
    spdlog::spdlog
    ${CMAKE_THREAD_LIBS_INIT}
)
