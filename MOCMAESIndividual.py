import math
import numpy as np
import MOCMAES

class MOCMAESIndividual:
    def __init__(self, x, problem, succProbability, sigma, evolProbability, c):
        self.x = x
        self.successRate = succProbability
        self.evolutionRate = evolProbability
        self.sigma = sigma
        self.problem = problem
        self.fitness = problem(self.x)
        self.c = c
        self.inputDim = x.size
        self.outputDim = self.fitness.size

    def isBetter(self, individual):
        isBetter = False
        for i in range(self.outputDim):
            if self.fitness[i] > individual.fitness[i]:
                return False
            if self.fitness[i] < individual.fitness[i]:
                isBetter = True
        return isBetter
    
    def updateStep(self):
        if self.successful:
            self.successRate = (1 - MOCMAES.Config.TARGET_SUCCESS_PROBABILITY) \
                                    * self.successRate + MOCMAES.Config.TARGET_SUCCESS_PROBABILITY
        else:
            self.successRate = (1 - MOCMAES.Config.TARGET_SUCCESS_PROBABILITY) \
                                    * self.successRate
        self.sigma = self.sigma * math.exp(self.successRate - MOCMAES.Config.TARGET_SUCCESS_PROBABILITY) \
                    / (MOCMAES.Config.DAMPING * (1 - MOCMAES.Config.TARGET_SUCCESS_PROBABILITY))
        
    def updateCovariance(self):
        if self.successRate < MOCMAES.Config.THRESHOLD:
            self.evolutionRate = (1 - MOCMAES.Config.CUMULATION_TIME_HORIZON) * self.evolutionRate \
                                + math.sqrt(MOCMAES.Config.CUMULATION_TIME_HORIZON \
                                * (2 - MOCMAES.Config.CUMULATION_TIME_HORIZON)) * self.step
            self.c = (1 - MOCMAES.Config.COVARIANCE_RATE) * self.c + MOCMAES.Config.COVARIANCE_RATE \
                    * (np.transpose(self.evolutionRate) * self.evolutionRate)
        else:
            self.evolutionRate = (1 - MOCMAES.Config.CUMULATION_TIME_HORIZON) * self.evolutionRate
            self.c = (1 - MOCMAES.Config.COVARIANCE_RATE) * self.c + MOCMAES.Config.COVARIANCE_RATE \
                    * (np.transpose(self.evolutionRate) * self.evolutionRate + MOCMAES.Config.CUMULATION_TIME_HORIZON * \
                       (2 - MOCMAES.Config.CUMULATION_TIME_HORIZON) * self.c)
            
    def mutate(self):
        x = np.random.multivariate_normal(self.x, pow(self.sigma, 2) * self.c)
        mutation = MOCMAESIndividual(x, self.problem, self.successRate, self.sigma, self.evolutionRate, self.c)
        mutation.step = (mutation.x - self.x) / self.sigma
        mutation.successful = mutation.isBetter(self)
        self.successful = mutation.successful
        return mutation