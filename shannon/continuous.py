'''
information.py
'''
__all__ = ['entropy', 'symbols_to_prob', 'combine_symbols', 'mi', 'cond_mi', 'mi_chain_rule']
import numpy as np
import pdb 

def entropy(data=None, prob=None, method='nearest-neighbors', bins=None, errorVal=1e-5):
    '''
    given a probability distribution (prob) or an interable of symbols (data) compute and
    return its continuous entropy.

    inputs:
    ------
        data:       samples by dimensions ndarray

        prob:       iterable with probabilities

        method:     'nearest-neighbors', 'gaussian', or 'bin'

        bins:       either a list of num_bins, or a list of lists containing 
                    the bin edges
        
        errorVal:   if prob is given, 'entropy' checks that the sum is about 1.
                    It raises an error if abs(sum(prob)-1) >= errorVal

        Different Methods:

        'nearest-neighbors' computes the binless entropy (bits) of a random vector
        using average nearest neighbors distance (Kozachenko and Leonenko, 1987).
        For a review see Beirlant et al., 2001 or Chandler & Field, 2007.

        'gaussian' computes the binless entropy based on estimating the covariance
        matrix and assuming the data is normally distributed.

        'bin' discretizes the data and computes the discrete entropy.
        
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
    
    if data:
        num_samples    = data.shape[0]
        num_dimensions = data.shape[1]

    if method == 'nearest-neighbors':
        from scipy.spatial.distance import pdist, squareform
        from scipy.special import gamma

        def getrho(x):
            '''
            Helper function for nearest-neighbors entropy. Returns the nearest
            neighbor distance in N dimensions.
            '''
            if len(x.shape) < 2:
                x = np.reshape(x, (x.shape[0],1))
            D = squareform(pdist(x, 'euclidean'))
            D = D + np.max(D)*eye(D.shape[0])
            return np.min(D, axis=0)

        if data is None:
            raise ValueError('Nearest neighbors entropy requires original data')

        if len(data.shape) > 1:
            k = num_dimensions
        else:
            k = 1
        
        Ak  = (k*pi**(float(k)/float(2)))/gamma(float(k)/float(2)+1)
        rho = getrho(data)
        
        # 0.577215... is the Euler-Mascheroni constant
        return k*mean(log2(rho)) + log2(num_samples*Ak/k) + log2(e)*0.5772156649

    elif method == 'gaussian':
        from numpy.linalg import det

        if data is None:
            raise ValueError('Nearest neighbors entropy requires original data')

        detCov = det(data.dot(data.transpose()))
        normalization = (2*pi*e)**num_dimensions
        
        return 0.5*np.log(normalization*detCov)

    elif method == 'bin':
        if prob is None and bins is None:
            raise ValueError('Either prob or bins must be specified.')

        if data is not None:
            prob = symbols_to_prob(data, bins=bins)
    
        # compute the log2 of the probability and change any -inf by 0s
        logProb = np.log2(prob)
        logProb[logProb==-np.inf] = 0
    
        # return sum of product of logProb and prob 
        # (not using np.dot here because prob, logprob are nd arrays)
        return -1.0* sum(prob * logProb)


def symbols_to_prob(data, bins=None, tol=10e-5):
    '''

    Return the probability distribution of symbols. Only probabilities are returned and in random order, 
    you don't know what the probability of a given label is but this can be used to compute entropy

    input:
        data:     ndarray of shape (samples, dimensions)
        bins:     either list of num_bins, or list of list of bin edges
        tol:      tolerance for determining if probabilities sum to 1

    returns:
        prob:     returns list of 1-d np arrays each containing probability of discretized symbols
    '''
    dimensionality = data.shape[1]
    if len(bins) != dimensionality:
        raise ValueError("Data dimensionality is %d but you only specified bins for %d dimensions."%(dimensionality, len(bins)))

    prob = np.histogramdd(data, bins, normed=True)

    if abs(sum(prob) - 1) > tol:
        raise ValueError("Probabilities should sum to 1, but actually sum to %f."%(sum(prob)))

    return prob



def mi(x, y, bins_x=None, bins_y=None, binx_xy=None, method='nearest-neighbors'):
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


    HX  = entropy(data=x, bins=bins_x, method=method)
    HY  = entropy(data=y, bins=bins_y, method=method)
    HXY = entropy(data=np.concatenate([x,y],axis=1), bins=bins_xy, method=method)

    return HX + HY - HXY


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
    probXZ = symbols_to_prob(combine_symbols(x, z))
    probYZ = symbols_to_prob(combine_symbols(y, z))
    probXYZ =symbols_to_prob(combine_symbols(x, y, z))
    probZ = symbols_to_prob(z)

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
    
