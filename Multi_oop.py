import matplotlib.pyplot as plt
import os
import random
from copy import deepcopy
from random import randint
import individual as vm
import numpy as np
import threading
from sklearn.model_selection import train_test_split
import sys


class MultiObjGP(threading.Thread):
    def __init__(self, filename):
        threading.Thread.__init__(self)
        self.current_file = filename
        self.pareto_front = []
        self.best_fitness = []
        self.best_fitness_2 = []
        self.avg_fitness = []
        self.avg_fitness_2 = []
        self.individual = []
        self.best_individual = None
        self.outfile = "output/" + self.current_file.split('/')[1] + ".txt"
        print (self.outfile + " submitted")
        self.test = False

    def run(self):
        self.train(self.current_file)

    def createPopulation(self, features, targets):
        individual = []

        for i in range(0, 100):
            vir_mach = vm.vm(targets)
            individual.append(vir_mach)

        return individual

    def readFile(self, filename):
        f = open(filename, 'r')
        data = f.readlines()

        shuttle_features = []
        line_count = 0

        try:
            for line in data:
                if (line_count == 0):
                    line_count = line_count + 1
                    continue
                line = line.strip()
                set = line.split(',')
                for i in range(0, len(set) - 1):
                    if (float(set[i]) == 0):
                        set[i] = float(set[i]) + 1
                set[-1] = float(set[-1])
                shuttle_features.append(set)
                # shuttle_target.append(set[9])

        except Exception as e:
            print e.message

        return shuttle_features

    def splitData(self, filename):

        try:
            shuttle_features = self.readFile(filename)
            shuttle_target = []

            for i in range(0, len(shuttle_features)):
                shuttle_target.append(shuttle_features[i][-1])

            X = np.array(shuttle_features)
            Y = np.array(shuttle_target)

            X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=99)

        except Exception as e:
            print e.message

        return X_train, X_test, y_train, y_test

    def shuttle_data(self, filename):

        try:
            X_train, X_test, y_train, y_test = self.splitData(filename)

            shuttle_features = []
            shuttle_features_test = []

            for set in X_train.tolist():
                temp_set = []
                for val in set:
                    temp_set.append(float(val))
                shuttle_features.append(temp_set)

            for set in X_test.tolist():
                temp_set = []
                for val in set:
                    temp_set.append(float(val))
                shuttle_features_test.append(temp_set)

            shuttle_features.sort(key=lambda x: x[-1])

            # Making indexList in the form [start,endIndex]
            shuttle_target = []
            indexList = []
            prevClass = shuttle_features[0][-1]
            indexList.append([])
            indexList[0].append(0)
            count = 0

            for i in range(0, len(shuttle_features)):
                shuttle_target.append(shuttle_features[i][-1])
                if (prevClass != shuttle_features[i][-1]):
                    indexList[count].append(i - 1)
                    indexList.append([])
                    count = count + 1
                    indexList[count].append(i)
                    prevClass = shuttle_features[i][-1]

            indexList[count].append(len(shuttle_features) - 1)

        except Exception as e:
            print e.message

        return shuttle_features, shuttle_features_test, shuttle_target, indexList

    def choseData(self, dataset, no_of_classes, indexes):

        try:
            lists = []
            sample = []

            for i in range(0, no_of_classes):  # change 2 to number of classes -1
                lists.append(dataset[indexes[i][0]: indexes[i][1]])

            lastIndex = []
            for i in range(0, 2 * no_of_classes + 1):
                lastIndex.append(0)

            # lastIndex = [0, 0, 0, 0, 0, 0, 0]
            max = 0
            for i in range(0, len(indexes)):
                temp = indexes[i][1] - indexes[i][0]
                if (temp > max):
                    max = temp

            if (max > 20000):
                max = max / 50
            else:
                max = 100
            #print max

            for i in range(0, max):
                list1 = []

                for j in range(0, no_of_classes):  # change 2 to number of classes -1
                    list2 = self.select(lists[j], lastIndex[j])
                    list1 = list1 + list2
                    lastIndex[j] = (lastIndex[j] + 100) % len(lists[j])

                sample = sample + list1

        except Exception as e:
            print e.message

        return sample

    # Method to calculate the fitness of individuals in a population
    def getFitness(self, population, features, targets, switch):
        class_count = []
        for i in range(0, len(population[0].classes)):
            class_count.append(0)

        for f in targets:
            class_count[int(f - 1)] = class_count[int(f - 1)] + 1
            '''
            if (f == 1.0):
                class_count[0] = class_count[0] + 1
            if (f == 2.0):
                class_count[1] = class_count[1] + 1
            if (f == 3.0):
                class_count[2] = class_count[2] + 1
            '''
        if(self.test):
            print class_count

        count = 0
        for newIndividual in population:
            m = 0
            newIndividual.fitness = 0
            newIndividual.init_multiFitness()
            for feature in features:
                try:
                    if (m < len(features) - 1):
                        t = targets[m]
                        max = newIndividual.fetch_decode_execute(feature, t)
                        m = m + 1

                        for i in range(0, len(newIndividual.classes)):

                            if (max == newIndividual.defalut_registers[i] and t == newIndividual.classes[i]):
                                newIndividual.multiFitness[i] = newIndividual.multiFitness[i] + 1
                        if (self.test):
                            if (max == 'R0' and t == 1.0):
                                newIndividual.true_positive = newIndividual.true_positive + 1
                                continue
                            if (max == 'R1' and t == 2.0):
                                newIndividual.true_negative = newIndividual.true_negative + 1
                                continue
                            if (max == 'R0' and t == 2.0):
                                newIndividual.false_negative = newIndividual.false_negative + 1
                                continue
                            if (max == 'R1' and t == 1.0):
                                newIndividual.false_positive = newIndividual.false_positive + 1



                except Exception, e:
                    print e.message
            try:
                for i in range(0, len(newIndividual.classes)):
                    if (class_count[i] != 0):
                        newIndividual.multiFitness[i] = float(
                            (float(newIndividual.multiFitness[i]) / class_count[i]) * 100)
                        newIndividual.multiFitness[i] = round(newIndividual.multiFitness[i], 2)

            except ZeroDivisionError:
                print features
                print 'Zero Division not allowed'

            count = count + 1
            # newIndividual.fitness= (newIndividual.fitness / len(features)) * 100
        r = 1

    def select(self, dataset, sIndex):
        data = []
        index = sIndex
        for i in range(0, 50):
            data.append(dataset[sIndex])
            sIndex = (sIndex + 1) % len(dataset)

        return data

    def __lt__(self, individual1, individual2):

        try:
            a = 0

            for i in range(0, len(individual1.classes)):
                temp1 = individual1.multiFitness[i]
                temp2 = individual2.multiFitness[i]
                if (temp1 <= temp2):
                    a = 1
                else:
                    a = -1
                    break

            if (a):
                for i in range(0, len(individual1.classes)):
                    temp1 = individual1.multiFitness[i]
                    temp2 = individual2.multiFitness[i]

                    if (temp1 < temp2):
                        a = 1
                        break
                    else:
                        a = -1

            # if(individual1.multiFitness[0] < 70 and individual1.multiFitness[1] < 70):

            if (a == False):
                if (individual2.multiFitness[0] >= 70 and individual2.multiFitness[1] >= 40):
                    return 1
                if (individual2.multiFitness[0] >= 60 and individual2.multiFitness[1] >= 40):
                    return 1

        except Exception as e:
            print e.message

        return a

    def getFileData(self, filename):

        try:
            self.current_file = filename
            shuttle_features, shuttle_features_test, shuttle_target, indexList = self.shuttle_data(filename)

            dataset = self.choseData(shuttle_features, len(set(shuttle_target)), indexList)
            targets = []

            #print indexList

            for i in range(0, len(dataset)):
                targets.append(dataset[i][-1])

            targets_test = []

            for i in range(0, len(shuttle_features_test)):
                targets_test.append(shuttle_features_test[i][-1])

        except Exception as e:
            print e.message

        return dataset, shuttle_features_test, targets, targets_test

    def makeGenDataset(self, no_of_classes, data):
        dataset_gen = []
        targets_gen = []
        temp_list = []
        temp_targets = []
        for i in range(0, len(data)):
            temp_list.append(data[i])
            if ((i + 1) % (no_of_classes * 400) == 0):
                random.shuffle(temp_list)
                for i in range(0, len(temp_list)):
                    temp_targets.append(temp_list[i][-1])

                dataset_gen.append(temp_list)
                targets_gen.append(temp_targets)
                temp_list = []
                temp_targets = []

        return dataset_gen, targets_gen

    # Method for mutation. This has not been used for assignment 1.
    def mutate(self, ind_list):
        for individual in ind_list:
            count = 1
            limit = randint(0, 5)
            # limit=1
            while (count <= limit):
                column = randint(0, 3)
                # column = 3
                value = randint(0, individual.no_of_instructions - 1)
                count = count + 1

                chance = randint(0, 1)

                if (chance):
                    individual.mode[value] = randint(0, 1)

                chance = randint(0, 1)
                if (chance):
                    individual.target[value] = randint(0, len(individual.classes) - 1)

                chance = randint(0, 1)
                if (chance):
                    individual.operator[value] = randint(0, 6)

                chance = randint(0, 1)

                if (individual.no_of_instructions > 10):
                    if (chance):
                        del individual.operator[value]
                        del individual.mode[value]
                        del individual.target[value]
                        del individual.source_ip[value]
                        individual.no_of_instructions = individual.no_of_instructions - 1

                '''
                if (column > -1):
                    #individual.mode[value] = randint(0, 1)
                    individual.target[value] = randint(0, len(individual.classes) - 1)
                    individual.operator[value] = randint(0, 3)
                '''

    def replaceInd(self, individuals):
        rng = len(individuals) * 0.2

        count = 0

        listcopy = []
        taken = []
        index = randint(0, int(len(individuals) * 0.8))

        for i in range(0, int(rng)):
            while (index in taken):
                index = randint(0, int(len(individuals) * 0.8))
            taken.append(index)
            listcopy.append(deepcopy(individuals[index]))
            count = count + 1

        self.mutate(listcopy)
        count = 0

        for i in range(0, int(rng)):
            del individuals[-1]

        # individuals = individuals + listcopy
        return individuals + listcopy

    def evolve(self, generations, data, targets):

        dataset_gen, targets_gen = self.makeGenDataset(len(self.individuals[0].classes), data)

        count_data = 0
        pareto_gen = 0
        # print dataset_gen[0]

        self.getFitness(self.individuals, dataset_gen[0], targets_gen[0], False)
        # print (len(dataset_gen))
        # graphData(individuals)

        for i in range(0, generations):
            try:
                if ((i + 1) % 5 == 0):
                    if (count_data == len(dataset_gen) - 1):
                        count_data = 0
                    else:
                        count_data = count_data + 1

                for j in range(0, len(self.individuals)):
                    self.individuals[j].dominance_count = 0
                    self.individuals[j].dominance_rank = 0

                self.individuals.sort()
                self.individuals.sort(key=lambda x: x.dominance_count)
                # individuals.sort(key=lambda x: x.dominance_rank)
                self.individuals.reverse()

                # self.best_fitness.append(self.individuals[0].multiFitness[0])
                # self.best_fitness_2.append(self.individuals[0].multiFitness[1])
                # self.avg_fitness.append(getAverage(individuals, 0))
                # self.avg_fitness_2.append(getAverage(individuals, 1))


                if (self.individuals[0].dominance_count >= 100):
                    print len(self.individuals)

                #print ("generation " + str(i) + "  " + str(self.individuals[0].dominance_count) + " " + str(
                #    self.individuals[0].multiFitness))

                # sys.stdout.write('.')

                self.individuals = self.replaceInd(self.individuals)
                # getFitness(children, dataset, targets, False)

                # getFitness([individuals[0]],dataset_gen[0],targets_gen[0], False)

                # individuals = individuals + children
                self.getFitness(self.individuals, dataset_gen[count_data], targets_gen[count_data], False)
                # getFitness(individuals, data, targets, False)
            except Exception, e:
                print e.message

        # individuals.sort(key=lambda x: x.dominance_rank)

        # printGraph(best_fitness, avg_fitness)
        # printGraph(best_fitness_2, avg_fitness_2)

        self.individuals.sort(key=lambda x: x.dominance_count)
        self.individuals.reverse()
        return self.individuals

    def train(self, filename):

        shuttle_features, shuttle_features_test, targets, targets_test = self.getFileData(filename)

        self.individuals = self.createPopulation(shuttle_features, targets)

        f = open(self.outfile, 'w')

        print>> f, self.current_file
        print>> f, "Train"
        generations = len(shuttle_features) / 150
        # print generations
        self.best = self.evolve(generations, shuttle_features, targets)

        self.best_individual = self.best[0]

        print>> f, self.best_individual.multiFitness

        print>> f, "---------------------------------------------------------------"
        print>> f, "Test"

        self.test = True

        self.getFitness(self.best[0:10], shuttle_features_test, targets_test, False)

        for i in range(0, 10):
            print>> f, "-------------------------------"
            print>> f, self.best[i].multiFitness

        print>> f, "true positive : " + str(self.best_individual.true_positive) + "       False positive: " + str(
            self.best_individual.false_positive)
        print>> f, "true negative : " + str(self.best_individual.true_negative) + "       False negative: " + str(
            self.best_individual.false_negative)

        for i in range(0, self.best[0].no_of_instructions):
            print>> f, (
                str(self.best[0].mode[i]) + "   " + str(self.best[0].target[i]) + "  " + str(
                    self.best[0].operator[i]) + "  " + str(
                    self.best[0].source_ip[i]))

        print "file " + self.current_file + " completed"
        print "------------------------------------------------------------------------------------------------------------"

        f.close()
