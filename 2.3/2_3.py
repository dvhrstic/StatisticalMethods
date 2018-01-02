#!/usr/bin/python3

import random
import collections
import numpy as np
import matplotlib.pyplot as plt
#import train as tr

import ipdb


observedSignals = 0
adjMatrix = 0
vertexSettings = 0
realSignals = 0
prob_sum =0

def blocks_gibbs_sampling(n):
    #ipdb.set_trace()
    #first evaluate the value of the vertex switch settings given at the beginning
    foo = [1.0, 2.0]
    new_values = False

    probability_first_settings = prob_sum
    propabilities = []
    settings = []
    #now change the values of the first element in the switch settings 
    propabilities.append(probability_first_settings)
    settings.append(vertexSettings)

    #do gibbs sampling 1000 times to get more values
    if n==2:
        print("value n = 2")
        for i in range(1000): 
            vertex = 0
            #ipdb.set_trace()
            while vertex <= 4:
                while new_values==False:
                    new_setting_value1 = random.choice(foo) #choose random new value for the settings
                    new_setting_value2 = random.choice(foo) #choose random new value for the settings

                    if vertexSettings[vertex] != new_setting_value1 and vertexSettings[vertex+1] != new_setting_value2:
                        break

                vertexSettings[vertex] = new_setting_value1
                vertexSettings[vertex+1] = new_setting_value2

                newSettingsValues2 = list(vertexSettings)

                adjMatrix = add_switch_settings_to_matrix() #on modifie la matrice avec les settings 
                prob_sum_intermediaire, probabilities, highest_prob = compute_probabilities_stop_position() #on calcule la nouvelle probabilité
                    
                propabilities.append(prob_sum_intermediaire) # on garde la nouvelle probabilité
                settings.append(newSettingsValues2) # on stocke l'échantillon
                vertex = vertex + 2
    
    if n==3:
        print("value n = 3")
        for i in range(1000): 
            vertex = 0
            #ipdb.set_trace()
            while vertex <= 3:
                while new_values==False:
                    new_setting_value1 = random.choice(foo) #choose random new value for the settings
                    new_setting_value2 = random.choice(foo) 
                    new_setting_value3 = random.choice(foo) 
                    if vertexSettings[vertex] != new_setting_value1 and vertexSettings[vertex+1] != new_setting_value2 and vertexSettings[vertex+2] != new_setting_value3:
                        break

                vertexSettings[vertex] = new_setting_value1
                vertexSettings[vertex+1] = new_setting_value2
                vertexSettings[vertex+2] = new_setting_value3

                newSettingsValues3 = list(vertexSettings)


                adjMatrix = add_switch_settings_to_matrix() #on modifie la matrice avec les settings 
                prob_sum_intermediaire, probabilities, highest_prob = compute_probabilities_stop_position() #on calcule la nouvelle probabilité
                    
                propabilities.append(prob_sum_intermediaire) # on garde la nouvelle probabilité
                settings.append(newSettingsValues3) # on stocke l'échantillon
                vertex = vertex + 3

    if n==6:
        print("value n = 6")
        for i in range(1000): 
            vertex = 0
            #ipdb.set_trace()
            while new_values==False:
                new_setting_value1 = random.choice(foo) #choose random new value for the settings
                new_setting_value2 = random.choice(foo) 
                new_setting_value3 = random.choice(foo) 
                new_setting_value4 = random.choice(foo)
                new_setting_value5 = random.choice(foo) 
                new_setting_value6 = random.choice(foo) 
                if vertexSettings[vertex] != new_setting_value1 and vertexSettings[vertex+1] != new_setting_value2 and vertexSettings[vertex+2] != new_setting_value3 and vertexSettings[vertex+3] != new_setting_value4 and vertexSettings[vertex+4] != new_setting_value5 and vertexSettings[vertex+5] != new_setting_value6:
                    break

            vertexSettings[vertex] = new_setting_value1
            vertexSettings[vertex+1] = new_setting_value2
            vertexSettings[vertex+2] = new_setting_value3
            vertexSettings[vertex+3] = new_setting_value4
            vertexSettings[vertex+4] = new_setting_value5
            vertexSettings[vertex+5] = new_setting_value6

            newSettingsValues6 = list(vertexSettings)

            adjMatrix = add_switch_settings_to_matrix() #on modifie la matrice avec les settings 
            prob_sum_intermediaire, probabilities, highest_prob = compute_probabilities_stop_position() #on calcule la nouvelle probabilité
                
            propabilities.append(prob_sum_intermediaire) # on garde la nouvelle probabilité
            settings.append(newSettingsValues6) # on stocke l'échantillon

    return propabilities, settings

def changeVertexSettingsValues(n, settings, new):
    set = settings[:]
    set[n] = new
    
    return set

def gibbs_sampling():
    #ipdb.set_trace()
    #first evaluate the value of the vertex switch settings given at the beginning
    foo = [1.0, 2.0]
    print(random.choice(foo))

    new_values = False

    probability_first_settings = prob_sum
    propabilities = []
    settings = []
    #now change the values of the first element in the switch settings 
    #propabilities.append(probability_first_settings)
    #settings.append(vertexSettings)

    #do gibbs sampling 1000 times to get more values
    for i in range(1000): 

        for vertex in range(6):
            #ipdb.set_trace()
            while new_values==False:
                new_setting_value = random.choice(foo) #choose random new value for the settings

                if vertexSettings[vertex] != new_setting_value:
                    break

            vertexSettings[vertex] = new_setting_value
            #newSettingsValues = changeVertexSettingsValues(vertex,vertexSettings,new_setting_value)
            newSettingsValues = list(vertexSettings)

            adjMatrix = add_switch_settings_to_matrix() #on modifie la matrice avec les settings 
            prob_sum_intermediaire, probabilities, highest_prob = compute_probabilities_stop_position() #on calcule la nouvelle probabilité
                
            propabilities.append(prob_sum_intermediaire) # on garde la nouvelle probabilité
            settings.append(newSettingsValues) # on stocke l'échantillon
            

    return propabilities, settings

def add_switch_settings_to_matrix():
    #add to the matrix the vertex settings on the diagonal
    j = 0
    for i in range(6):
        adjMatrix[i, j] = vertexSettings[i]
        j = j + 1

    return adjMatrix


def compute_c(s, t):
    #ipdb.set_trace()
    if t == 0:
        return 1 / 6

    adjacent_vertex_1 = 0
    adjacent_vertex_2 = 0
   
    next_i = 0

    for i in range(6):
        value = adjMatrix[s[0] , i]
        if value != 0 and i != s[1][0] and i != s[1][1]:
            adjacent_vertex_1 = i
            next_i = i
            break
    for i in range(next_i + 1, 6):
        value = adjMatrix[s[0] , i]
        if value != 0 and i != s[1][0] and i != s[1][1] and i != adjacent_vertex_1:
            adjacent_vertex_2 = i
            break

    incident_edge_1 = (adjacent_vertex_1, s[0] )
    incident_edge_2 = (adjacent_vertex_2, s[0] )

    edge_value = adjMatrix[s[1][0], s[1][1]]
    incident_edge_1_value = adjMatrix[s[0] , adjacent_vertex_1] 
    vertex_switch_value = adjMatrix[s[0] , s[0] ]

    observations = observedSignals[t - 1]
    previous_step = t - 1

    next_stop_1 = (adjacent_vertex_1, incident_edge_1)
    next_stop_2 = (adjacent_vertex_2, incident_edge_2)


    if edge_value == 3 and observations == 3:
        return (compute_c(next_stop_1, previous_step) + compute_c(next_stop_2, previous_step)) * (1.0 - 0.05)

    elif edge_value == 3 and observations != 3:
        return (compute_c(next_stop_1, previous_step) + compute_c(next_stop_2, previous_step)) * 0.05

    elif edge_value == 1 and vertex_switch_value == 1 and observations == 1 and incident_edge_1_value == 3:
        return compute_c(next_stop_1, previous_step) * (1.0 - 0.05)

    elif edge_value == 1 and vertex_switch_value == 1 and observations != 1 and incident_edge_1_value == 3:
        return compute_c(next_stop_1, previous_step) * 0.05

    elif edge_value == 2 and vertex_switch_value == 2 and observations == 2 and incident_edge_1_value == 3:
        return compute_c(next_stop_1, previous_step) * (1.0 - 0.05)

    elif edge_value == 2 and vertex_switch_value == 2 and observations != 2 and incident_edge_1_value == 3:
        return compute_c(next_stop_1, previous_step) * 0.05

    elif edge_value == 1 and vertex_switch_value == 2:
        return 0.0

    elif edge_value == 2 and vertex_switch_value == 1:
        return 0.0

    return 0.0

def compute_probabilities_stop_position(): 
    #ipdb.set_trace()
    total_probabilities = 0.0
    probabilities = []
    steps = 5
    i = 0

    #for each vertices find all edges deg(v) =3 in order to calculate the probability for all states
    for vertex in range(6): 
        prob = 0.0
        #find the first adjacent vertex
        for adjacent_vertex in range(6): 
            if adjMatrix[vertex, adjacent_vertex] != 0 and adjacent_vertex != vertex: #if there is a connection between two different vertices  
                i = adjacent_vertex #save the last adjacent vertex value postion in the matrix
                break

        prob = compute_c((vertex, (vertex, adjacent_vertex)), steps) #call the method which compute the probability of these vertex going to the other by the edge
        probabilities.append(prob) # add the probability to the tab
        total_probabilities = total_probabilities + prob # sum each position propabilities to Calculates p(s | G, sigma)

        #find the second adjacent vertex
        for adjacent_vertex in range(i + 1, 6):
            if adjMatrix[vertex, adjacent_vertex] != 0 and adjacent_vertex != vertex:
                i = adjacent_vertex
                break

        prob = compute_c((vertex, (vertex, adjacent_vertex)), steps)
        probabilities.append(prob)
        total_probabilities = total_probabilities + prob

        #find the third adjacent vertex
        for adjacent_vertex in range(i + 1, 6):
            if adjMatrix[vertex, adjacent_vertex] != 0 and adjacent_vertex != vertex:
                break

        prob = compute_c((vertex, (vertex, adjacent_vertex)), steps)
        probabilities.append(prob)
        total_probabilities = total_probabilities + prob

    highest_probabilities = max(probabilities)

    return (total_probabilities, probabilities, highest_probabilities)



if __name__ == '__main__':

    #create the undirected graph model with the vertex settings
    #adjMatrix, vertexSettings = tr.createExperiment(n=2, saveOpt=True, displayOpt=False)
    #print(adjMatrix)
    #print("")

    #print("vertex settings :", vertexSettings)

    #create the HMM observations and simulate the train
    #observedSignals, realSignals = tr.simulate(adjMatrix, vertexSettings, numIter=5, pErr=0.05, saveOpt=True, displayOpt=False)

    #print("obseved signal :" ,observedSignals)
    #print("real signals : ",realSignals)
    #print("")

    #take data from text

    adjMatrix = np.loadtxt('exp_2_adjacency.txt', usecols=range(6), converters=None)
    vertexSettings = np.loadtxt('exp_2_vertexsettings.txt', usecols=range(1), converters=None)

    print(adjMatrix)
    print("vertex settings :", vertexSettings)

    observedSignals = np.loadtxt('simulate_numIter_5_pErr_0_startPos_4_startDir_1_observedsignals.txt', usecols=range(1), converters=None)
    realSignals = np.loadtxt('simulate_numIter_5_pErr_0_startPos_4_startDir_1_realsignals.txt', usecols=range(1), converters=None)

    #populate the matrix with the switch settings
    adjMatrix = add_switch_settings_to_matrix()
    print(adjMatrix)

    prob_sum, probabilities, highest_prob = compute_probabilities_stop_position()
    print("Correct settings probability", prob_sum)
    print("")

    print("List of all states probability", probabilities)
    print("")

    print("highest probability", highest_prob)

    #all_settings_probabilities, all_settings = gibbs_sampling()
    all_settings_probabilities, all_settings = blocks_gibbs_sampling(2)

    
    #print("all_settings_probabilities : ",all_settings_probabilities)
    print("")
    #print("all_settings : ",all_settings)

    index = all_settings_probabilities.index(max(all_settings_probabilities))
    print(" settings switches :", all_settings[index])
    print("probability with these settigns : ", all_settings_probabilities[index])
   




