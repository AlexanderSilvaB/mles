include_directories(../../lib)

find_package( OpenCV )

if(OpenCV_FOUND)

    set(SRC main.cpp)

    add_executable(tsp ${SRC})
    target_link_libraries(tsp mles)
    target_link_libraries( tsp ${OpenCV_LIBS} )
    
else()
    message(STATUS "OpenCV required to build tsp")
endif()