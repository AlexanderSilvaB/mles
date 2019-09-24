include_directories(../../lib)

find_package( OpenCV )

set(SRC main.cpp Pong.cpp)

add_executable(pong ${SRC})
target_link_libraries(pong  nnes)
target_link_libraries( pong ${OpenCV_LIBS} )