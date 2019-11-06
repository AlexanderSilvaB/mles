include_directories(..)

set(SRC NN.cpp QLearning.cpp GA.cpp DataSet.cpp Layer.cpp)

add_library(mles ${SRC})
target_link_libraries (mles eigen)
