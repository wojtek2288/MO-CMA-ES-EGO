import math
import numpy as np
import MOCMAESIndividual

class Config:
    GENERATIONS = 50
    POPULATION = 100
    SIGMA = 5.0
    MAX_CROWDING_DISTANCE = pow(10,2)
    THRESHOLD = 0.44
    TARGET_SUCCESS_PROBABILITY = pow(math.sqrt(0.5) + 5, -1)
    SUCCESS_RATE_AVG = TARGET_SUCCESS_PROBABILITY / (2 + TARGET_SUCCESS_PROBABILITY)
    DAMPING = 1 + POPULATION / 2
    CUMULATION_TIME_HORIZON =  2 / (2 + POPULATION)
    COVARIANCE_RATE = 2 / (pow(POPULATION, 2) + 6)

class MOCMAES:
    def run(self, problem, x0):
        dim = x0.size
        population = []

        for _ in range(Config.POPULATION):
            x = np.random.rand(dim) * 4 * Config.SIGMA \
                - 2 * Config.SIGMA
            identityArray = np.identity(dim)
            individual = MOCMAESIndividual.MOCMAESIndividual(
                x,
                problem,
                Config.TARGET_SUCCESS_PROBABILITY,
                Config.SIGMA,
                0,
                identityArray)
            population.append(individual)

        for _ in range(Config.GENERATIONS):
            currentPopulation = []
            for i in range(Config.POPULATION):
                currentPopulation.append(population[i].mutate())

            for i in range(Config.POPULATION):
                population[i].updateStep()
                currentPopulation[i].updateStep()
                currentPopulation[i].updateCovariance()
                currentPopulation.append(population[i])

            population = self.select(currentPopulation)

        return population[0].x
    
    def select(self, individuals):
        individualsCount = len(individuals)

        distances = self.crowdingDistances(individuals)
        ranks = np.zeros(individualsCount)

        for i in range(individualsCount):
            for j in range(individualsCount):
                if individuals[i].isBetter(individuals[j]):
                    ranks[j] += 1

        perm = ranks.argsort()
        ranks = ranks[perm]
        individuals = np.array(individuals)[perm]

        nextGenerationCount = 0
        nextGeneration = []

        for i in range(2 * Config.POPULATION):
            tempIndividuals = []
            tempDistances = []

            while (ranks[nextGenerationCount] == i and nextGenerationCount < Config.POPULATION):
                tempIndividuals.append(individuals[nextGenerationCount])
                tempDistances.append(distances[nextGenerationCount])
                nextGenerationCount += 1
            
            if (len(tempIndividuals) > 0):
                tempIndividuals = np.array(tempIndividuals)
                tempDistances = np.array(tempDistances)

                perm = tempDistances.argsort()
                tempIndividuals = tempIndividuals[perm]
                tempDistances = tempDistances[perm]

                for individual in tempIndividuals:
                    nextGeneration.append(individual)
        
        return nextGeneration
     

    def crowdingDistances(self, individuals):
        individualsCount = len(individuals)
        dim = individuals[0].fitness.size
        distances = np.zeros(individualsCount)
        fitnessValuesOfDimensions = np.zeros(individualsCount)

        for i in range(dim):
            for j in range(individualsCount):
                fitnessValuesOfDimensions[j] = individuals[j].fitness[i]
            
            perm = fitnessValuesOfDimensions.argsort()
            fitnessValuesOfDimensions = fitnessValuesOfDimensions[perm]

            distances[perm[0]] = Config.MAX_CROWDING_DISTANCE
            distances[perm[individualsCount - 1]] = Config.MAX_CROWDING_DISTANCE

            for j in range(2, individualsCount - 1):
                distances[perm[j]] += (fitnessValuesOfDimensions[i + 1] - fitnessValuesOfDimensions[i - 1]) \
                                    / (fitnessValuesOfDimensions[individualsCount - 1] - fitnessValuesOfDimensions[0])
        
        return distances