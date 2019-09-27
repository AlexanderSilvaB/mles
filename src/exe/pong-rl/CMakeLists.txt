include_directories(../../lib)

find_package( OpenCV )

if(OpenCV_FOUND)

    include_directories(../pong)

    set(SRC main.cpp ../pong/Pong.cpp)

    add_executable(pong-rl ${SRC})
    target_link_libraries(pong-rl  mles)
    target_link_libraries( pong-rl ${OpenCV_LIBS} )

else()
    message(STATUS "OpenCV required to build pong-rl")
endif()