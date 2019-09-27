include_directories(../../lib)

set(SRC main.cpp)

add_executable(csv ${SRC})
target_link_libraries(csv  mles)