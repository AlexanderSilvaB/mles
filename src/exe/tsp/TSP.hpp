#ifndef _TSP_HPP_
#define _TSP_HPP_

#include <mles/GA.hpp>
#include <iostream>
#include "City.hpp"

class TSP : public mles::GAEnv
{
    private:
        int N;
    public:
        TSP(int n = 25) : mles::GAEnv()
        {
            N = n;
        }

        void createInitialSequence()
        {
            for(int n = 0; n < N; n++)
            {
                City city(std::rand() % 400, std::rand() % 400);
                addToInitial(&city);
            }
        }

        double routeDistance(mles::GASequence& route)
        {
            double pathDistance = 0;
            for(int i = 0; i < route.size(); i++)
            {
                City* from = route[i]->asPtr<City>();
                City* to;
                if(i + 1 < route.size())
                    to = route[i + 1]->asPtr<City>();
                else
                    to = route[0]->asPtr<City>();
                pathDistance += from->distance(*to);
            }
            return pathDistance;
        }

        double fitness(mles::GASequence& route)
        {
            return 1.0f / routeDistance(route);
        }
};


#endif