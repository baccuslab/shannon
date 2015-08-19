"""
Implementing a new idea, from x, y (iterables with discrete symbols)
How much MI does it cost to loose a symbol?
loop through all the symbols in x and for each symbol, replace all its instances by other values
in x (drawn from the the same probability probribution) and recompute the MI.
"""
__all__ = ['toy_example', 'remove_symbol_from_dist', 'inhibit_symbol', 'change_response', 
    'differentiate_mi']

from shannon import discrete
#import shannon
import numpy as np
import matplotlib.pyplot as plt
import pdb

def remove_symbol_from_dist(dist, index):
    '''
    prob is a ndarray representing a probability distribution.
    index is a number between 0 and and the number of symbols ( len(prob)-1 )
    return the probability distribution if the element at 'index' was no longer available
    '''
    if type(dist) is not Distribution:
        raise TypeError("remove_symbol_from_dist got an object ot type {0}".format(type(dist)))

    new_prob = dist.prob.copy()
    new_prob[index]=0
    new_prob /= sum(new_prob)
    return Distribution(new_prob)


def change_response(x, prob, index):
    '''
    change every response in x that matches 'index' by randomly sampling from prob
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
    dist = Distribution(discrete.symbols_to_prob(x))

    diff = np.zeros(len(dist.prob))

    for i in range(len(dist.prob)):
        i = int(i)
        dist = Distribution(remove_symbol_from_dist(dist, i).prob)

        new_x = change_response(x, dist, i)

        diff[i] = discrete.mi(x,y)

    return diff

class Distribution:
    def __init__(self, prob):
        
        if type(prob) is not np.ndarray:
            raise TypeError('Distribution requires an ndarray as its unique parameter')

        self.prob = prob

        self.cumsum = prob.cumsum()

    def sample(self, *args):
        '''
        generate a random number in [0,1) and return the index into self.prob
        such that self.prob[index] <= random_number but self.prob[index+1] > random_number

        implementation note: the problem is identical to finding the index into self.cumsum
        where the random number should be inserted to keep the array sorted. This is exactly
        what searchsorted does. 

        usage:
            myDist = Distribution(array(0.5, .25, .25))
            x = myDist.sample()         # generates 1 sample
            x = myDist.sample(100)      # generates 100 samples
            x = myDist.sample(10,10)    # generates a 10x10 ndarray
        '''
        return self.cumsum.searchsorted(np.random.rand(*args))
        
if __name__ == "__main__":
    x, y, diff = toy_example()


