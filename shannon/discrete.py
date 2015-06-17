'''
information.py
'''
__all__ = ['entropy', 'symbols_to_prob', 'combine_symbols', 'mi', 'cond_mi', 'mi_chain_rule']
import numpy as np
import pdb 

from collections import Counter
class Counter(Counter):
    def prob(self):
        return np.array(list(self.values()))

def entropy(data=None, prob=None, errorVal=1e-5):
    '''
    given a probability distribution (prob) or an interable of symbols (data) compute and
    return its entropy

    inputs:
    ------
        data:       iterable of symbols

        prob:       iterable with probabilities
        
        errorVal:   if prob is given, 'entropy' checks that the sum is about 1.
                    It raises an error if abs(sum(prob)-1) >= errorVal
    '''
    
    #pdb.set_trace()
    if prob is None and data is None:
        raise ValueError("%s.entropy requires either 'prob' or 'data' to be defined"%(__name__))

    if prob is not None and data is not None:
        raise ValueError("%s.entropy requires only 'prob' or 'data to be given but not both"%(__name__))

    if prob is not None and not isinstance(prob, np.ndarray):
        raise TypeError("'entropy' in '%s' needs 'prob' to be an ndarray"%(__name__)) 

    if prob is not None and abs(prob.sum()-1) > errorVal:
        raise ValueError("parameter 'prob' in '%s.entropy' should sum to 1"%(__name__))
    

    if data is not None:
        prob = symbols_to_prob(data).prob()
    
    # compute the log2 of the probability and change any -inf by 0s
    logProb = np.log2(prob)
    logProb[logProb==-np.inf] = 0
    
    # return dot product of logProb and prob
    return -1.0* np.dot(prob, logProb)


def symbols_to_prob(symbols):
    '''
    Return a dict mapping symbols to  probability.

    input:
    -----
        symbols:     iterable of hashable items
                     works well if symbols is a zip of iterables
    '''
    myCounter = Counter(symbols)

    N = len(list(symbols))*1.0    # symbols might be a zip object in python 3
                                  # in python 2 N has to be a float since otherwise n/N folds back to integer
                                  # division

    for k in myCounter:
        myCounter[k] /= N

    return myCounter
    

def combine_symbols(*args):
    '''
    Combine different symbols into a 'super'-symbol

    args can be an iterable of iterables that support hashing

    see example for 2D ndarray input
    
    usage:
        1) combine two symbols, each a number into just one symbol
        x = numpy.random.randint(0,4,1000)
        y = numpy.random.randint(0,2,1000)
        z = combine_symbols(x,y)

        2) combine a letter and a number
        s = 'abcd'
        x = numpy.random.randint(0,4,1000)
        y = [s[randint(4)] for i in range(1000)]
        z = combine_symbols(x,y)

        3) suppose you are running an experiment and for each sample, you measure 3 different
        properties and you put the data into a 2d ndarray such that:
            samples_N, properties_N = data.shape
        
        and you want to combine all 3 different properties into just 1 symbol
        In this case you have to find a way to impute each property as an independent array
            
            combined_symbol = combine_symbols(*data.T)

        
        4) if data from 3) is such that:
            properties_N, samples_N  = data.shape
        
        then run:

            combined_symbol = combine_symbols(*data)

    '''
    #pdb.set_trace()
    for arg in args:
        if len(arg)!=len(args[0]):
            raise ValueError("combine_symbols got inputs with different sizes")

    return tuple(zip(*args))


def mi(x, y):
    '''
    compute and return the mutual information between x and y
    
    inputs:
    -------
        x, y:   iterables of hashable items
    
    output:
    -------
        mi:     float

    Notes:
    ------
        if you are trying to mix several symbols together as in mi(x, (y0,y1,...)), try
                
        info[p] = _info.mi(x, info.combine_symbols(y0, y1, ...) )
    '''
    #pdb.set_trace()
    # dict.values() returns a view object that has to be converted to a list before being
    # converted to an array
    # the following lines will execute properly in python3, but not python2 because there
    # is no zip object
    try:
        if isinstance(x, zip):
            x = list(x)
        if isinstance(y, zip):
            y = list(y)
    except:
        pass

    probX = symbols_to_prob(x).prob()
    probY = symbols_to_prob(y).prob()
    probXY = symbols_to_prob(combine_symbols(x, y)).prob()

    return entropy(prob=probX) + entropy(prob=probY) - entropy(prob=probXY)

def cond_mi(x, y, z):
    '''
    compute and return the mutual information between x and y given z, I(x, y | z)
    
    inputs:
    -------
        x, y, z:   iterables with discrete symbols
    
    output:
    -------
        mi:     float

    implementation notes:
    ---------------------
        I(x, y | z) = H(x | z) - H(x | y, z)
                    = H(x, z) - H(z) - ( H(x, y, z) - H(y,z) )
                    = H(x, z) + H(y, z) - H(z) - H(x, y, z)
    '''
    #pdb.set_trace()
    # dict.values() returns a view object that has to be converted to a list before being converted to an array
    probXZ = symbols_to_prob(combine_symbols(x, z)).prob()
    probYZ = symbols_to_prob(combine_symbols(y, z)).prob()
    probXYZ =symbols_to_prob(combine_symbols(x, y, z)).prob()
    probZ = symbols_to_prob(z).prob()

    return entropy(prob=probXZ) + entropy(prob=probYZ) - entropy(prob=probXYZ) - entropy(prob=probZ)

def mi_chain_rule(X, y):
    '''
    Decompose the information between all X and y according to the chain rule and return all the terms in the chain rule.
    
    Inputs:
    -------
        X:          iterable of iterables. You should be able to compute [mi(x, y) for x in X]

        y:          iterable of symbols

    output:
    -------
        ndarray:    terms of chaing rule

    Implemenation notes:
        I(X; y) = I(x0, x1, ..., xn; y)
                = I(x0; y) + I(x1;y | x0) + I(x2; y | x0, x1) + ... + I(xn; y | x0, x1, ..., xn-1)
    '''
    
    # allocate ndarray output
    chain = np.zeros(len(X))

    # first term in the expansion is not a conditional information, but the information between the first x and y
    chain[0] = mi(X[0], y)
    
    #pdb.set_trace()
    for i in range(1, len(X)):
        chain[i] = cond_mi(X[i], y, X[:i])
        
    return chain
    
def bin(x, bins, maxX=None, minX=None):
    '''
    bin signal x using 'binsN' bin. If minX, maxX are None, they default to the full 
    range of the signal. If they are not None, everything above maxX gets assigned to
    binsN-1 and everything below minX gets assigned to 0, this is effectively the same
    as clipping x before passing it to 'bin'

    input:
    -----
        x:      signal to be binned, some sort of iterable

        bins:   int, number of bins
                iterable, bin edges

        maxX:   clips data above maxX

        minX:   clips data below maxX

    output:
    ------
        binnedX:    x after being binned

        bins:       bins used for binning.
                    if input 'bins' is already an iterable it just returns the
                    same iterable

    example:
        # make 10 bins of equal length spanning from x.min() to x.max()
        bin(x, 10)      

        # use predefined bins such that each bin has the same number of points (maximize
        entropy)
        binsN = 10
        percentiles = list(np.arange(0, 100.1, 100/binsN)) 
        bins = np.percentile(x, percentiles)
        bin(x, bins)
    '''
    #pdb.set_trace()
    if maxX is None:
        maxX = x.max()

    if minX is None:
        minX = x.min()

    if not np.iterable(bins):
        bins = np.linspace(minX, maxX+1e-5, bins+1)
    
    # digitize works on 1d array but not nd arrays. 
    # So I pass the flattened version of x and then reshape back into x's original shape
    return np.digitize(x.flatten(), bins).reshape(x.shape), bins
