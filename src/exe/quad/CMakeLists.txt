include_directories(../../lib)

set(SRC main.cpp)

add_executable(quad ${SRC})
target_link_libraries(quad  nnes)