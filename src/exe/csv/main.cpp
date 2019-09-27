#include <iostream>
#include <mles/mles.hpp>

using namespace std;
using namespace mles;

int main(int argc, char *argv[])
{
    // Randomize
    std::srand((unsigned int) time(0));

    // Construct the NN model
    NN nn(2, 1);

    DataSet dataset = nn.createDataSet();
    if(dataset.fromCSV("data/xor.csv"))
    {
        dataset.print();
        dataset.save("xor");
        dataset.load("xor");
        dataset.print();
    }
        


    return 0;
}