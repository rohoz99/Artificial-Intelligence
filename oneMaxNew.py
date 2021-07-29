from numpy.random import rand
from numpy.random import randint
import matplotlib.pyplot as plt
from random import choice, random
import numpy as np

target = list("10101010000101010111")
charset = "01"
stringVals = "0"
#parent = [choice(charset) for _ in range(len(target))]
minmutaterate = 0.01
k = 6
children = range(100)
crossOverRate = 0.8
bitStringLen = 20
num_Generations = 1000
#0.75
gens = []
fitValArray = []
numGens = []
fitnessTotal = []
fitVal = 0
list = []
populationtot = 100
one_max = False
perfectfitness = float(len(target))
targetString = False
deceptiveLandscape = True

def fitness(trial):
    if one_max:
       test_list = map(int, trial)
       return sum(test_list)

    if targetString:
        return sum(t == h for t, h in zip(trial, target))

    if deceptiveLandscape:

        fitness = (trial.count('1') / bitStringLen) / 2
        if fitness == 0:
            return 1
        return fitness

   # else:
       # return (trial.count('1') / bitStringLen) / 2
        #
     #test_list = map(int, trial)
    #
 #   else:
     #   return (len(trial) * 2)


def mutateRate():

    return 1 - ((perfectfitness - fitness(parent)) / perfectfitness * (1 - minmutaterate))


def mutate(parent, rate):
    return [(ch if random() <= rate else choice(charset)) for ch in parent]


def createPopulation(population, popSize, fitnessFunc, bitStringLength):
    for i in range(popSize):
        bitString = ""
        for j in range(bitStringLength):
            bitString += str(randint(0, 1))
        individual = {'Bitstring': bitString, 'Fitness': fitnessFunc(bitString)}
        population.append(individual)


def run():
    fitPerc = fitness(parent) * 100. / perfectfitness
    fitnessVal = fitness(parent)
    fitValArray.append(fitnessVal)

    print("#%-4i Fitness: %4.1f%% '%s'" %
          (iterations,fitPerc, ''.join(parent)))


def crossover(a, b,r_cross):
    # one point cross over
    c1,c2 = a.copy(),b.copy()
    if rand() < r_cross:
        pt = randint(1, len(a) - 2)
        # perform crossover
        c1 = a[:pt] + b[pt:]
        c2 = a[:pt] + b[pt:]
    return c1, c2



p=0
parent = [choice(stringVals) for _ in range(len(target))]
iterations = 0
while p <= num_Generations:
 #

 center = int(len(children) / 2)
 #while parent != target:
 rate = mutateRate()
 #run()
 #iterations += 1
 #gens.append(iterations)
 if iterations == 0:
  run()
  gens.append(iterations)
  iterations+=1

 elif iterations !=0:
      copies = [mutate(parent, rate) for _ in children] + [parent]
      parent1 = max(copies[:center], key=fitness)
      parent2 = max(copies[center:], key=fitness)
      parent = max(crossover(parent1, parent2,crossOverRate), key=fitness)
      mutate(parent,rate)


      fitPerc = fitness(parent) * 100. / perfectfitness
      fitnessVal = fitness(parent)
      fitValArray.append(fitnessVal)
      print("#%-4i Fitness: %4.1f%% '%s'" %
            (iterations, fitPerc, ''.join(parent)))

      gens.append(iterations)
      iterations += 1

      if parent == target:
         numGens.append(iterations)
         list.append(tuple(fitValArray))
 p += 1




 fitnessTotal.append(sum(fitValArray))

# fitnessVal = np.empty(numGens.shape[0])

z= len(fitnessTotal) -1
gg = 0
while (z>0):
     fitnessTotal[z] = (((fitnessTotal[z] - fitnessTotal[z-1]) /populationtot) * 100)
     z+=-1
     if (z== 0):
         (fitnessTotal[z]  * 100)


'''
bestFitness = min(fitnessTotal)
indexBest = fitnessTotal.index(bestFitness)
bestNoGens = numGens[indexBest]
'''


plt.title("Graph for 1(c) Deceptive Landscape")
plt.ylabel(" Average Fitness ")
plt.xlabel("Generations")
'''
maxGens = int(np.max(numGens))
generationTotal = len(numGens)
print(generationTotal)
'''
fitHistoryMean = [np.mean(fitness) for fitness in list]
fitHistoryMax = [np.max(fitness) for fitness in list]
#fitHistoryMean[0] = fitHistoryMean[0]/100
#fitness_history_mean = [np.mean(fitnessTotal) for fitness in fitnessTotal]

avgfit = np.average(fitnessTotal)
avgGens = np.average(numGens)
print(list)
print(len(fitValArray))
plt.plot((range(num_Generations+1)),fitValArray)
plt.show()



