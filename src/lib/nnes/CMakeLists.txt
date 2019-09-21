include_directories(..)

find_package (Eigen3 3.3 REQUIRED NO_MODULE)

set(SRC nnes.cpp DataSet.cpp Layer.cpp)

add_library(nnes ${SRC})
target_link_libraries (nnes Eigen3::Eigen)
