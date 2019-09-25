include_directories(../../lib)

set(SRC main.cpp)

add_executable(xor ${SRC})
target_link_libraries(xor  mles)