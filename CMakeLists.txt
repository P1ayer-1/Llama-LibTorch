cmake_minimum_required(VERSION 3.26)
project(cppTorch)

set(CMAKE_CXX_STANDARD 17)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")


file(GLOB_RECURSE CPP_SOURCES "source/*.cpp")

add_executable(cppTorch ${CPP_SOURCES})



# add include directories
target_include_directories(cppTorch PRIVATE
        ${TORCH_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}/source/llama/public

        )


target_link_libraries(cppTorch "${TORCH_LIBRARIES}")

# log TORCH_INSTALL_PREFIX
message(STATUS "TORCH_INSTALL_PREFIX: ${TORCH_INSTALL_PREFIX}")
get_target_property(TARGET_LOCATION cppTorch BINARY_DIR)
message(STATUS "TARGET_LOCATION: ${TARGET_LOCATION}")
if (MSVC)
file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
add_custom_command(TARGET cppTorch
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${TORCH_DLLS}
        $<TARGET_FILE_DIR:cppTorch>)
endif (MSVC)
