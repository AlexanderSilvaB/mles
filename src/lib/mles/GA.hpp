#ifndef _MLES_GA_HPP_
#define _MLES_GA_HPP_

#include <vector>
#include <string>
#include <memory>
#include <stdarg.h>
#include <unordered_map>

namespace mles
{
    class GAGene
    {
        public:
            GAGene();
            virtual ~GAGene();

            virtual GAGene* copy();
            
            template<class T>
            T& as()
            {
                return (T&)this;
            }

            template<class T>
            T* asPtr()
            {
                return (T*)this;
            }

            virtual std::string toString();
    };

    typedef std::shared_ptr <GAGene > GAGenePtr;
    typedef std::vector < GAGenePtr > GASequence;
    typedef std::vector < GASequence > GAPopulation;
    typedef std::vector < std::pair<int, double> > GAFitness;
    typedef std::vector < double > GAProgress;
    typedef std::vector<int> GASelection;

    enum GASelectionMode { FitnessProportion, Tournament, Custom };

    typedef struct FittingSettings_st
    {
        int popSize, eliteSize, generations;
        double mutationRate;
        GASelectionMode selectionMode;
        bool enableElitism;

        FittingSettings_st(int popSize = 100, int eliteSize = 20, double mutationRate = 0.01f, int generations = 500)
        {
            this->popSize = popSize;
            this->eliteSize = eliteSize;
            this->mutationRate = mutationRate;
            this->generations = generations;
            this->selectionMode = GASelectionMode::FitnessProportion;
            this->enableElitism = true;
        }

    }FittingSettings;

    typedef struct
    {
        GAProgress progress;
        GASequence best;
    }FittingResults;

    class GAEnv
    {
        private:
            GASequence initialSequence;

        public:
            GAEnv();
            virtual ~GAEnv();

            void addToInitial(GAGene* gene);
            GASequence createSequence();
            
            virtual void createInitialSequence();
            virtual double fitness(GASequence& route);
            virtual GASelection selection(GAFitness& ranked);

    };

    class GA
    {
        private:
            GAEnv *env;
            FittingSettings settings;
            GAPopulation createPopulation();
            GAFitness rank(GAPopulation& population);
            static bool compareFitness(std::pair<int, double> a, std::pair<int, double> b);
            
            GASelection selection(GAFitness& ranked);
            GASelection selectionFitnessProportion(GAFitness& ranked);
            GASelection selectionTournament(GAFitness& ranked);

            GAPopulation matingPool(GAPopulation& population, GASelection& selectionResults);
            GASequence breed(GASequence& parent1, GASequence& parent2);
            GAPopulation breedPopulation(GAPopulation& pool);
            GASequence mutate(GASequence& individual);
            GAPopulation mutatePopulation(GAPopulation& population);
            GAPopulation nextGeneration(GAPopulation& currentGen);

        public:
            GA(GAEnv *env);
            virtual ~GA();

            FittingResults fit(const FittingSettings& settings);
            FittingResults fit();
    };
}

#endif