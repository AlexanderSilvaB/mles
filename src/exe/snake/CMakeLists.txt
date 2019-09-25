include_directories(../../lib)

find_package( OpenCV )

if(OpenCV_FOUND)

    set(SRC main.cpp Snake.cpp)

    add_executable(snake ${SRC})
    target_link_libraries(snake mles)
    target_link_libraries( snake ${OpenCV_LIBS} )
    
else()
    message(STATUS "OpenCV required to build snake")
endif()