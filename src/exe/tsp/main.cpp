#include <iostream>
#include <mles/mles.hpp>
#include "TSP.hpp"
#include <opencv2/opencv.hpp>
#include "plot.hpp"

using namespace std;
using namespace mles;
using namespace cv;

int main(int argc, char *argv[])
{
    std::srand((unsigned int) time(0));

    // GA
    TSP tsp(30);
    GA ga((GAEnv*)&tsp);

    FittingSettings settings;
    settings.generations = 1000;
    settings.selectionMode = FitnessProportion;
    FittingResults results = ga.fit(settings);
    
    cout << "Initial distance: " << results.progress.front() << endl;
    cout << "Final distance: " << results.progress.back() << endl;

    // Plot
    Mat route(400, 400, CV_8UC3);
    for(int i = 0; i < results.best.size(); i++)
    {
        cv::Point2d current(results.best[i]->asPtr<City>()->getX(), results.best[i]->asPtr<City>()->getY());
        circle(route, current, 5, Scalar(0,0,255),CV_FILLED);
        cv::Point2d next;
        if(i < results.best.size()-1)
        {
            next.x = results.best[i+1]->asPtr<City>()->getX();
            next.y = results.best[i+1]->asPtr<City>()->getY();
        }
        else
        {
            next.x = results.best[0]->asPtr<City>()->getX();
            next.y = results.best[0]->asPtr<City>()->getY();
        }
        line(route, current, next, Scalar(255,0,0), 1, CV_AA);
    }
    imshow("Route", route);
    waitKey();

    Mat plot_result;
    Mat data(results.progress, false);
    Ptr<plot::Plot2d> plot = plot::Plot2d::create(data);
    plot->setPlotBackgroundColor(Scalar(50, 50, 50)); 
    plot->setPlotLineColor(Scalar(50, 50, 255));
    plot->render(plot_result);          
    imshow("Graph", plot_result);
    waitKey();

    return 0;
}