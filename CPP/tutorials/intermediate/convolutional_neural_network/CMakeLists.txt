cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(convolutional-neural-network VERSION 1.0.0 LANGUAGES CXX)

# Add the profiling flags to both the compiler and linker flags
set(EXECUTABLE_NAME convolutional-neural-network)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
# set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -g")

add_executable(${EXECUTABLE_NAME})
target_sources(${EXECUTABLE_NAME} PRIVATE src/main.cpp
                                          src/convnet.cpp
                                          src/cifar10.cpp
                                          include/convnet.h
                                          include/cifar10.h
)


# Add FetchContent for nlohmann/json
include(FetchContent)
FetchContent_Declare(
  nlohmann_json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG v3.10.0  # Specify the version you want to use
)
FetchContent_MakeAvailable(nlohmann_json)

target_link_libraries(${EXECUTABLE_NAME} ${TORCH_LIBRARIES} image-io nlohmann_json::nlohmann_json)
target_include_directories(${EXECUTABLE_NAME} PRIVATE include)

set_target_properties(${EXECUTABLE_NAME} PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED YES
)

if(DOWNLOAD_DATASETS)
    add_dependencies(${EXECUTABLE_NAME} imagenette)
endif()

if(MSVC)
    set_target_properties(${EXECUTABLE_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_BINARY_DIR})
    set_target_properties(${EXECUTABLE_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_CURRENT_BINARY_DIR})
    include(copy_torch_dlls)
    copy_torch_dlls(${EXECUTABLE_NAME})
endif(MSVC)
