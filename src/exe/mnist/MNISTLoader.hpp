#ifndef _LOADER_HPP_
#define _LOADER_HPP_

#include <mles/DataSet.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <sstream>
#include <iostream>

using namespace mles;
using namespace Eigen;
using namespace std;
using namespace cv;

class MNISTLoader
{
    private:
        static Mat readDigit(const string& folder, int n, int digit)
        {
            stringstream path;
            path << folder << digit << "_" << n << ".png";
            return imread(path.str(), IMREAD_GRAYSCALE);
        }
    public:
        static DataSet Load(const string& folder, int N)
        {
            Mat img = readDigit(folder, 0, 0);
            int sz = img.cols * img.rows;

            DataSet data(sz, 10);

            for(int n = 0; n < N; n++)
            {
                for(int digit = 0; digit < 10; digit++)
                {
                    img = readDigit(folder, n, digit);
                    
                    VectorXd input = data.createInput();
                    for(int i = 0; i < sz; i++)
                        input[i] = img.data[i] / 255.0;

                    VectorXd output = data.createOutput();
                    output.setZero();
                    output[digit] = 1;

                    data.add(input, output);
                }
            }

            return data;
        }
};

#endif
