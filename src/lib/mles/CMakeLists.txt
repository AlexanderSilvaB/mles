include_directories(..)

find_package (Eigen3 3.3 REQUIRED NO_MODULE)

set(SRC NN.cpp QLearning.cpp DataSet.cpp Layer.cpp)

add_library(mles ${SRC})
target_link_libraries (mles Eigen3::Eigen)
