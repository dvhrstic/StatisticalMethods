
import numpy as np
from scipy import special
from collections import Counter
import functools
import ipdb

def generate_sequences():
    
    value = []
    for j in range(5):
        random_value = []
        for i in range(10):
            random_value.append(['A', 'B', 'C','D'][np.argmax(np.random.multinomial(1,np.random.dirichlet([1,1,1,1])))]) 
        value.append(random_value)

    pos = []
    for i in range(5):
        pos.append(np.random.randint(0, 10-3+1))

    
    sequence = []
    for s, p in zip(value, pos):
        value2 = []
        for i in range(len(s)):
            if(i >= p and i < p+3):
                value2.append(['A', 'B', 'C','D'][np.argmax(np.random.multinomial(1,np.random.dirichlet([12,7,3,1], 3)[i-p]))])
            else:
                value2.append(s[i])
        sequence.append(value2)

    return sequence, pos

def run(seq):
    

    position = []
    for i in range(5):
        position.append([np.random.randint(0, 10-3+1)]) 
    print(position)

    for i in range(200):
        for j in range(5):
            #probability = position_of_starting_point(seq, position, j)
            probability_of_position_of_motif = []
            for r_i in range(10-3):
                pos = list(position)
                print(pos)
                pos[j] = r_i

                #count number of time a letter appears in the background and in the magic word
                for i in range(len(seq)):
                    b = seq[i][:pos[i]] + seq[i][pos[i]+3:]
                    m = [seq[i][pos[i]:pos[i]+3]]

                #flatten the list
                count_b = Counter([item for sublist in b for item in sublist])
                count_m = [Counter([m[i][j] for i in range(len(seq))]) for j in range(3)]

                b_base = float(special.gamma(sum([1,1,1,1]))) / special.gamma(len(['A', 'B', 'C','D'])*(10-3) + sum([1,1,1,1]))
                m_base = float(special.gamma(sum([12,7,3,1]))) / special.gamma(len(['A', 'B', 'C','D'])+ sum([12,7,3,1]))

                probability = []

                for k in range(len(['A', 'B', 'C','D'])):
                    probability.append(float(special.gamma(count_b[['A', 'B', 'C','D'][k]] + [1,1,1,1][k])) / special.gamma([1,1,1,1][k]))

                probability_b = functools.reduce(operator.mul, probability, 1)
                prob_b = probability_b*b_base


                for i in range(3):
                    probability_m = []
                    for k in range(len(['A', 'B', 'C','D'])):
                        probability_m.append(float(special.gamma(count_m[i][['A', 'B', 'C','D'][k]] + [12,7,3,1][k])) / special.gamma([12,7,3,1][k]))
                        proba_m = functools.reduce(operator.mul, probability_m, 1)
                        prob_m = proba_m*b_base
                        probability_back *= prob_m

                probability_of_position_of_motif.append(probability_back)
            
    return position

if __name__ == "__main__":

    sequences, positions  = generate_sequences()
  
    print("sequences", sequences)
    print("positions", positions)

    samples = run(sequences)
    print(samples)
