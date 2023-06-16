import random
import numpy as np
import scipy as sci
from scipy.special import erf
from scipy.linalg import solve, cholesky

class Config:
    POPULATION = 20
    GENERATIONS = 50
    TOURNAMENT_SIZE = 5
    ENCODING = 20
    REPRODUCTION_PROBABILITY = 0.2
    CROSSOVER_PROBABILITY = 0.6
    MUTATION_PROBABILITY = 0.4
    MUTATION_RATE = 0.1
    BITS = MUTATION_RATE * ENCODING / MUTATION_PROBABILITY

class EGO():
    def __init__(self, nvars, X, Y):
        self.nvars = nvars
        self.X = X
        self.Y = Y

        self.UpperTheta = np.ones((2, )) * 0.0
        self.LowerTheta = np.ones((2, )) * -3.0
        self.y_best = min(self.Y)

    def next(self):
        x, EI = self.run(self.func, np.zeros((self.nvars, )), np.ones((self.nvars, )))
        return x, EI

    def func(self, x):
        y_hat, SSqr = self.predict(x)

        if SSqr != 0:
            EI = (self.y_best - y_hat) * (0.5 + 0.5 * erf((self.y_best - y_hat) / np.sqrt(2 * SSqr))) + \
                np.sqrt(0.5 * SSqr / np.pi) * np.exp(-0.5 * (self.y_best - y_hat) ** 2 / SSqr)
            EI = -EI
        else:
            EI = 0

        return EI

    def train(self):
        self.Theta, self.MinNegLnLik = self.run(self.likelihood, self.LowerTheta, self.UpperTheta)
        self.U, self.mu, self.SigmaSqr = self.likelihood(self.Theta)[1:]

    def likelihood(self, theta):
        X = self.X
        f = self.Y
        Theta = 10 ** theta
        n = np.size(X, 0)
        one = np.ones((n, 1))
        R = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                R[i, j] = np.exp(-sum(Theta * sci.power(abs(X[i, :] - X[j, :]), 2)))

        R = R + R.T + np.eye(n) + np.eye(n) * np.finfo(np.float32).eps

        U = cholesky(R)

        LnDetPsi = 2 * sum(np.log(abs(np.diag(U))))
        mu = (np.dot(one.T, solve(U, solve(U.T, f)))) / (np.dot(one.T, solve(U, solve(U.T, one))))
        SigmaSqr = (np.dot((f - one * mu).T, solve(U, solve(U.T, f - one * mu)))) / n
        NegLnLike = -1 * (-(n / 2) * np.log(SigmaSqr) - .5 * LnDetPsi)

        return NegLnLike, U, mu, SigmaSqr

    def predict(self, x):
        X = self.X
        f = self.Y
        Theta = 10 ** self.Theta
        n = np.size(X, 0)
        one = np.ones((n, 1))
        U = self.U

        r = np.zeros((n, 1))
        for i in range(n):
            r[i] = np.exp(-sum(Theta * sci.power(abs(X[i, :] - x), 2)))

        y_hat = self.mu + np.dot(r.T, solve(U, solve(U.T, f - one * self.mu)))
        SSqr = abs(self.SigmaSqr * (1 - np.dot(r.T, solve(U, solve(U.T, r)))))
        return y_hat, SSqr

    def run(self, func, LB, UB):
        population = []
        populationFitness = []

        for _ in range(Config.POPULATION):
            ind = self.randInd()
            fit = self.fitness(func, self.decode(ind, LB, UB))

            population.append(ind)
            populationFitness.append(self.fitness(func, self.decode(ind, LB, UB)))

        for _ in range(Config.GENERATIONS):
            newPopulation = []
            newFitness = []
            
            bestFit = min(populationFitness)
            newPopulation.append(population[populationFitness.index(bestFit)])
            newFitness.append(bestFit)
            
            for _ in range(1, Config.POPULATION):
                op = self.flip(Config.REPRODUCTION_PROBABILITY, Config.CROSSOVER_PROBABILITY)

                if op == 0:
                    ind = self.tournament(population, populationFitness)
                    newPopulation.append(population[ind])
                    newFitness.append(populationFitness[ind])
                elif op == 1:
                    parent1 = self.tournament(population, populationFitness)
                    parent2 = self.tournament(population, populationFitness)

                    children = self.crossover(population[parent1], population[parent2])
                    child1Fitness = self.fitness(func, self.decode(children[0], LB, UB))
                    child2Fitness = self.fitness(func, self.decode(children[1], LB, UB))

                    newPopulation.append(children[0])
                    newFitness.append(child1Fitness)
                    newPopulation.append(children[1])
                    newFitness.append(child2Fitness)
                else:
                    toMutate = self.tournament(population, populationFitness)
                    mutatedIndividual = self.mutation(population[toMutate])
                    fit = self.fitness(func, self.decode(mutatedIndividual, LB, UB))

                    newPopulation.append(mutatedIndividual)
                    newFitness.append(fit)
            
            while len(population) != len(newPopulation):
                worst = max(newFitness)
                newPopulation.remove(newPopulation[newFitness.index(worst)])   
                newFitness.remove(worst)
            
            population = newPopulation
            populationFitness = newFitness
            
        return np.array(self.decode(population[populationFitness.index(min(populationFitness))], LB, UB)), min(populationFitness)
                
    def tournament(self, population, populationFitness):
        pool = []
        poolFit = []
        
        for _ in range(Config.TOURNAMENT_SIZE):
            rand = int(random.random() * Config.POPULATION)
            pool.append(population[rand])
            poolFit.append(populationFitness[rand])

        return poolFit.index(min(poolFit))  

    def crossover(self, individual1, individual2):
        i = int(random.random() * Config.ENCODING * self.nvars)
        first = individual1[0:i] + individual2[i:]
        second = individual2[0:i] + individual1[i:]
        return [first, second]

    def mutation(self, ind):   
        idx = int(random.random() * Config.ENCODING * self.nvars)
        if ind[idx] == '0':
            newIdx = ind[0:idx - 1] + '1' + ind[idx:]
        else:
            newIdx = ind[0:idx - 1] + '0' + ind[idx:]
        return newIdx
            
    def decode(self, genotype, LB, UB):
        phenotype = []
        for i in range(self.nvars):
            decode = int(genotype[Config.ENCODING * i:Config.ENCODING * (i + 1)], 2)
            gene = LB[i] + (decode) * (UB[i] - LB[i]) / (pow(2, Config.ENCODING) - 1)
            phenotype.append(gene)    

        return phenotype
        
    def fitness(self, func, phenotype):
        return func(np.array(phenotype))[0]

    def randInd(self):
        ind = []
        for _ in range(Config.ENCODING * self.nvars):
            if random.random() > .5:
                ind.append('1')
            else:
                ind.append('0')
                
        return ''.join(ind)

    def flip(self, reproductionProbability, crossoverProbability):
        i = random.random()
        
        if i < reproductionProbability:
            num = 0
        elif i < reproductionProbability + crossoverProbability and i >= reproductionProbability:
            num = 1
        else:
            num = 2

        return num