#include "GA.hpp"
#include <algorithm>
#include <iostream>

using namespace std;
using namespace mles;

// Gene
GAGene::GAGene()
{

}

GAGene::~GAGene()
{

}

GAGene* GAGene::copy()
{
    return new GAGene();
}

std::string GAGene::toString()
{
    return "gene";
}

// Env
GAEnv::GAEnv()
{

}

GAEnv::~GAEnv()
{

}

void GAEnv::addToInitial(GAGene* gene)
{
    GAGene* copy = gene->copy();
    initialSequence.push_back( GAGenePtr(copy) );
}

GASequence GAEnv::createSequence()
{
    GASequence sequence = initialSequence;
    random_shuffle ( sequence.begin(), sequence.end() );
    return sequence;
}

void GAEnv::createInitialSequence()
{

}

GASelection GAEnv::selection(GAFitness& ranked)
{
    GASelection result(ranked.size());
    for(int i = 0; i < ranked.size(); i++)
    {
        result.push_back(ranked[i].first);
    }
    return result;
}

double GAEnv::fitness(GASequence& route)
{
    return 0;
}

// GA
GA::GA(GAEnv* env)
{
    this->env = env;
}

GA::~GA()
{

}

GAPopulation GA::createPopulation()
{
    GAPopulation population(settings.popSize);
    for(int i = 0; i < settings.popSize; i++)
    {
        population[i] = env->createSequence();
    }
    return population;
}

bool GA::compareFitness(pair<int, double> a, pair<int, double> b) 
{
    return a.second > b.second;
}

GAFitness GA::rank(GAPopulation& population)
{
    GAFitness fitness(population.size());
    for(int i = 0; i < population.size(); i++)
    {
        fitness[i] = pair<int, double>(i, env->fitness(population[i]));
    }
    std::sort(fitness.begin(), fitness.end(), compareFitness);
    return fitness;
}

GASelection GA::selection(GAFitness& ranked)
{
    switch(settings.selectionMode)
    {
        case FitnessProportion:
            return selectionFitnessProportion(ranked);
        case Tournament:
            return selectionTournament(ranked);
        case Custom:
            return env->selection(ranked);
    }
    return selectionFitnessProportion(ranked); 
}

GASelection GA::selectionFitnessProportion(GAFitness& ranked)
{
    GASelection result;
    vector<double> cumsum(ranked.size());
    double sum = 0;

    for(int i = 0; i < ranked.size(); i++)
    {
        if(settings.enableElitism && i < settings.eliteSize)
            result.push_back(ranked[i].first);
        if(i == 0)
            cumsum[i] = ranked[i].second;
        else
            cumsum[i] = cumsum[i - 1] + ranked[i].second;
        sum += ranked[i].second;
    }

    sum = 1.0f / sum;

    for(int i = 0; i < cumsum.size(); i++)
    {
        cumsum[i] = 100 * cumsum[i] * sum;
    }
    
    double pick;
    int len = ranked.size();
    if(settings.enableElitism)
        len -= settings.eliteSize;

    for(int i = 0; i < len; i++)
    {
        pick = rand() % 100;
        for(int j = 0; j < ranked.size(); j++)
        {
            if(pick <= cumsum[j])
            {
                result.push_back(ranked[i].first);
                break;
            }
        }
    }

    return result;
}

GASelection GA::selectionTournament(GAFitness& ranked)
{
    GASelection result;
    vector<double> cumsum(ranked.size());
    double sum = 0;

    if(settings.enableElitism)
    {
        for(int i = 0; i < ranked.size(); i++)
        {
            result.push_back(ranked[i].first);
        }
    }

    int pickI, pick;
    int len = ranked.size();
    if(settings.enableElitism)
        len -= settings.eliteSize;

    for(int i = 0; i < len; i++)
    {

        GASelection selectedIndex;
        GAFitness selected;
        while(selectedIndex.size() < settings.eliteSize)
        {  
            pickI = rand() % ranked.size();
            pick = ranked[pickI].first;
            if(find(selectedIndex.begin(), selectedIndex.end(), pick) == selectedIndex.end())
            {
                selectedIndex.push_back(pick);
                selected.push_back(ranked[pickI]);
            }
        }
        std::sort(selected.begin(), selected.end(), compareFitness);
        result.push_back(selected[0].first);
    }

    return result;
}

GAPopulation GA::matingPool(GAPopulation& population, GASelection& selectionResults)
{
    GAPopulation newPop(selectionResults.size());
    int index;
    for(int i = 0; i < selectionResults.size(); i++)
    {
        index = selectionResults[i];
        newPop[i] = population[index];
    }
    return newPop;
}

GASequence GA::breed(GASequence& parent1, GASequence& parent2)
{
    int geneA = rand() % parent1.size();
    int geneB = rand() % parent1.size();
    int startGene = min(geneA, geneB);
    int endGene = max(geneA, geneB);

    GASequence child;

    for(int i = startGene; i < endGene; i++)
    {
        child.push_back(parent1[i]);
    }

    for(int i = 0; i < parent2.size(); i++)
    {
        if(find(child.begin(), child.end(), parent2[i]) == child.end())
        {
            child.push_back(parent2[i]);
        }
    }
    return child;
}

GAPopulation GA::breedPopulation(GAPopulation& pool)
{
    int len = pool.size() - settings.eliteSize;
    GAPopulation mpool = pool;

    GAPopulation children(pool.size());

    random_shuffle ( mpool.begin(), mpool.end() );

    int i = 0;
    for(i = 0; i < settings.eliteSize; i++)
    {
        children[i] = pool[i];
    }

    for(int j = 0; j < len; j++, i++)
    {
        children[i] = breed(mpool[j], mpool[ pool.size() - j - 1 ]);
    }
    return children;
}

GASequence GA::mutate(GASequence& individual)
{
    double random;
    int swapWith;
    GAGenePtr gene1, gene2;

    GASequence retIndividual = individual;

    for(int swapped = 0; swapped < retIndividual.size(); swapped++)
    {
        random = (rand() % 100) / 0.01f;
        if(random < settings.mutationRate)
        {
            swapWith = rand() % retIndividual.size();
            gene1 = retIndividual[swapped];
            gene2 = retIndividual[swapWith];
            retIndividual[swapped] = gene2;
            retIndividual[swapWith] = gene1;
        }
    }

    return retIndividual;
}

GAPopulation GA::mutatePopulation(GAPopulation& population)
{
    GAPopulation mutatedPop(population.size());
    for(int ind = 0; ind < population.size(); ind++)
    {
        mutatedPop[ind] = mutate( population[ind] );
    }
    return mutatedPop;
}

GAPopulation GA::nextGeneration(GAPopulation& currentGen)
{
    GAFitness popRanked = rank(currentGen);
    GASelection selectionResults = selection(popRanked);
    GAPopulation matingpool = matingPool(currentGen, selectionResults);
    GAPopulation children = breedPopulation(matingpool);
    GAPopulation next = mutatePopulation(children);
    return next;
}

FittingResults GA::fit()
{
    env->createInitialSequence();
    GAPopulation pop = createPopulation();

    FittingResults results;

    results.progress.push_back( 1.0f / rank(pop)[0].second );

    for (int i = 0; i < settings.generations; i++)
    {
        pop = nextGeneration(pop);
        results.progress.push_back( 1.0f / rank(pop)[0].second );
    }

    int bestRouteIndex = rank(pop)[0].first;
    results.best = pop[bestRouteIndex];

    return results;
}

FittingResults GA::fit(const FittingSettings& settings)
{
    this->settings = settings;
    return fit();
}