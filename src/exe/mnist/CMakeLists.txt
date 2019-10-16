include_directories(../../lib)

find_package( OpenCV )

if(OpenCV_FOUND)

    set(SRC main.cpp)

    add_executable(mnist ${SRC})
    target_link_libraries(mnist  mles)
    target_link_libraries( mnist ${OpenCV_LIBS} )

else()
    message(STATUS "OpenCV required to build mnist")
endif()