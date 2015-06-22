"""
How much MI costs loosing a symbol?
Implement a new idea, from x, y (iterables with discrete symbols)
loop through all the symbols in x and for each symbol, replace all its instances by other values
in x (drawn from the the same probability distribution) and recompute the MI.
"""
__all__ = ['toy_example', 'remove_sample_from_prob', 'inhibit_symbol', 'change_response', 
    'differentiate_mi']

import Information.information as info
from Information.information import Distribution
import numpy as np
import matplotlib.pyplot as plt
import pdb

def remove_sample_from_prob(prob, index):
    '''
    prob is a ndarray representing a probability distribution.
    index is a number between 0 and len(prob)-1
    return the probability distribution if the element at 'index' was no longer available
    '''
    new_prob = prob[:]
    new_prob[index]=0
    return new_prob/sum(new_prob)

def inhibit_symbol(myDist, index):
    """
    return a Districtibution object that has the same probability distribution as
    in 'myDist' affter suppressing the symbol at 'index'
    """
    #pdb.set_trace()

    return info.Distribution(remove_sample_from_prob(myDist.prob, index))

def change_response(x, dist, index):
    '''
    change every response in x that matches 'index' by randomly sampling from dist
    '''
    #pdb.set_trace()
    N = (x==index).sum()
    #x[x==index]=9
    x[x==index] = dist.sample(N)

def toy_example():
    """
    Make a toy example where x is uniformly distributed with N bits and y
    follows x but with symbol dependent noise.
    x=0 -> y=0
    x=1 -> y=1 + e
    x=2 -> y=2 + 2*e
    ...
    x=n -> y=n + n*e
    where by n*e I am saying that the noise grows
    """
    #pdb.set_trace()
    N=4
    m = 100
    x = np.zeros(m*(2**N))
    y = np.zeros(m*(2**N))

    for i in range(1, 2**N):
        x[i*m:(i+1)*m] = i
        y[i*m:(i+1)*m] = i + np.random.randint(0, 2*i, m)

    diff = differentiate_mi(x,y)
    return x, y, diff

def differentiate_mi(x, y):
    '''
    for each symbol in x, change x such that there are no more of such symbols
    (replacing by a random distribution with the same proba of all other symbols)
    and compute mi(new_x, y)
    '''
    #pdb.set_trace()
    dist = info.Distribution(info.labels_to_prob(x))

    diff = np.zeros(len(dist.prob))

    for i in range(len(dist.prob)-1):
        #pdb.set_trace()
        i = int(i)
        dist = info.Distribution(inhibit_symbol(dist, i).prob)

        new_x = change_response(x, dist, i)

        diff[i] = info.mi(x,y)

    return diff

if __name__ == "__main__":
    x, y, diff = toy_example()


