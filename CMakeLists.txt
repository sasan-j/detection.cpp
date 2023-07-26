cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(app)

enable_language(CUDA)

find_package(Torch REQUIRED)
find_package(CUDAToolkit REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")


include_directories(include)

file(GLOB SOURCES "app.cpp" "src/*.cpp" "src/*.cu")
add_executable(app ${SOURCES})

target_link_libraries(app "${TORCH_LIBRARIES}")

set_property(TARGET app PROPERTY CXX_STANDARD 17)
set_property(TARGET app PROPERTY CUDA_ARCHITECTURES 75)

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET app
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:app>)
endif (MSVC)