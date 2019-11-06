#ifndef _CITY_HPP_
#define _CITY_HPP_

#include <mles/GA.hpp>
#include <iostream>
#include <cmath>
#include <sstream>

class City : public mles::GAGene
{
    private:
        double x, y;
    public:
        City(double x, double y) : mles::GAGene()
        {
            this->x = x;
            this->y = y;
        }

        double distance(const City& city)
        {
            double dx = x - city.x;
            double dy = y - city.y;
            return sqrt( dx*dx + dy*dy );
        }

        double getX()
        {
            return x;
        }

        double getY()
        {
            return y;
        }

        mles::GAGene* copy()
        {
            City* city = new City(x, y);
            return city;
        }

        std::string toString()
        {
            std::stringstream ss;
            ss << x << "," << y;
            return ss.str();
        }
};


#endif