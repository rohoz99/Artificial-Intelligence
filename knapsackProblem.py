
from random import randint
import matplotlib.pyplot as plt
import numpy as np
import random as rd
import sys


numberOfItems = np.arange(1,11)
weights = [18, 9, 23, 20, 59, 61, 70, 75, 76, 30]
values = [78, 35, 89, 36, 94, 75, 74, 79, 80, 16]
knapsack_capacity = 156    #Maximum weight that the knapsack can hold

gens = 50
pop_size = (8, 10)
firstPop = np.random.randint(2, size = pop_size)
firstPop = firstPop.astype(int)


def fitnessFunc(weights, values, population, threshold):
    fitnessVal = np.empty(population.shape[0])
    for i in range(population.shape[0]):
        check1 = np.sum(population[i] * values)
        check2 = np.sum(population[i] * weights)
        if check2 <= threshold:
            fitnessVal[i] = check1
        else :
            fitnessVal[i] = 0
           # print(fitnessVal)
    return fitnessVal.astype(int)

def crossover(parents, numChildren):
    offspring = np.empty((numChildren, parents.shape[1]))
    crossPoint = int(parents.shape[1]/2)
    crossRate = 0.45
    i=0
    while (parents.shape[0] < numChildren):
        parent1 = i%parents.shape[0]
        parent2 = (i+1)%parents.shape[0]
        x = rd.random()
        if x > crossRate:
            continue
        parent1 = i%parents.shape[0]
        parent2 = (i+1)%parents.shape[0]
        offspring[i,0:crossPoint] = parents[parent1,0:crossPoint]
        offspring[i,crossPoint:] = parents[parent2,crossPoint:]
        i=+1
    return offspring



def selection(fitness, num_parents, population):
    fitness = list(fitness)
    parents = np.empty((num_parents, population.shape[1]))
    for i in range(num_parents):
        highFitness = np.where(fitness == np.max(fitness))
        parents[i,:] = population[highFitness[0][0], :]
        fitness[highFitness[0][0]] = -999999
    return parents




def mutation(offsprings):
    rateOfMutation = 0.06
    mutated = np.empty((offsprings.shape))
    for i in range(mutated.shape[0]):
        ranVal = rd.random()
        mutated[i,:] = offsprings[i,:]
        if ranVal > rateOfMutation:
            continue
        randomValue = randint(0,offsprings.shape[1]-1)
        if mutated[i,randomValue] == 0 :
            mutated[i,randomValue] = 1
        else :
            mutated[i,randomValue] = 0
    return mutated


def optimize(weight, value, population, pop_size, num_generations, threshold):
    params, fitness_history = [], []
    numParents = int(pop_size[0] / 2)
    num_offsprings = pop_size[0] - numParents
    i=0
    for i in range(num_generations):
        fitness = fitnessFunc(weight, value, population, threshold)
        fitness_history.append(fitness)
        parents = selection(fitness, numParents, population)
        offsprings = crossover(parents, num_offsprings)
        mutated = mutation(offsprings)
        population[0:parents.shape[0], :] = parents
        population[parents.shape[0]:, :] = mutated

    lastGenFitness = fitnessFunc(weight, value, population, threshold)
    print('Last generations fitness: \n{}\n'.format(lastGenFitness))
    maxFit = np.where(lastGenFitness == np.max(lastGenFitness))
    params.append(population[maxFit[0][0], :])
    return params, fitness_history,lastGenFitness



parameters, fitHistory,lastGenFit = optimize(weights, values, firstPop, pop_size, gens, knapsack_capacity)
print('Optimized Params: \n{}'.format(parameters))
selected_items = numberOfItems * parameters
indexSelected = []
select = 0
print('\nItems which can be entered into the knapsack:')
for i in range(selected_items.shape[1]):
  if selected_items[0][i] != 0:
     select = selected_items[0][i] - 1
     indexSelected.append(select)
     print('{}\n'.format(selected_items[0][i]))


#print(fitHistory)
fitHistoryMean = [np.mean(fitness) for fitness in fitHistory]
print(fitHistoryMean)
fitHistoryMax = [np.max(fitness) for fitness in fitHistory]
print(fitHistoryMax)
plt.plot(list(range(gens)), fitHistoryMax, label = 'Max Fitness')
plt.plot(list(range(gens)), fitHistoryMean, label = 'Mean Fitness')
plt.legend()
plt.title('Fitness v Generations Knapsack Q Part (b)')
plt.xlabel('Generations')
plt.ylabel('Fitness')
plt.show()
print(np.asarray(fitHistory).shape)


selectedWeights = []
for i in range(len(indexSelected)):
    idx = indexSelected[i]
    selectedWeights.append(weights[idx])

cap = str(knapsack_capacity)
print("\n------------- Printing Results for Knapsack of size: " +cap+" -----------------")
print("Best Fitness:",max(lastGenFit))
print("Best Weight:",sum(selectedWeights))
print('Optimized Contents:\n',parameters)
