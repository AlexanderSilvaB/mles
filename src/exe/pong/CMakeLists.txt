include_directories(../../lib)

find_package( OpenCV )

if(OpenCV_FOUND)

    set(SRC main.cpp Pong.cpp)

    add_executable(pong ${SRC})
    target_link_libraries(pong  mles)
    target_link_libraries( pong ${OpenCV_LIBS} )

else()
    message(STATUS "OpenCV required to build pong")
endif()