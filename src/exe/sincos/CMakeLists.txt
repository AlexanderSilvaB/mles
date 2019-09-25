include_directories(../../lib)

set(SRC main.cpp)

add_executable(sincos ${SRC})
target_link_libraries(sincos  mles)