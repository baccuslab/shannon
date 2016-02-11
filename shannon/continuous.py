'''
information.py
'''
__all__ = ['entropy', 'symbols_to_prob', 'mi', 'cond_entropy']
import numpy as np


def entropy(data=None, prob=None, method='nearest-neighbors', bins=None, errorVal=1e-5, units='bits'):
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

        units:      either 'bits' or 'nats'

        Different Methods:

        'nearest-neighbors' computes the binless entropy (bits) of a random vector
        using average nearest neighbors distance (Kozachenko and Leonenko, 1987).
        For a review see Beirlant et al., 2001 or Chandler & Field, 2007.

        'gaussian' computes the binless entropy based on estimating the covariance
        matrix and assuming the data is normally distributed.

        'bin' discretizes the data and computes the discrete entropy.

    '''

    if prob is None and data is None:
        raise ValueError("%s.entropy requires either 'prob' or 'data' to be defined" % __name__)

    if prob is not None and data is not None:
        raise ValueError("%s.entropy requires only 'prob' or 'data to be given but not both" % __name__)

    if prob is not None and not isinstance(prob, np.ndarray):
        raise TypeError("'entropy' in '%s' needs 'prob' to be an ndarray" % __name__)

    if prob is not None and abs(prob.sum()-1) > errorVal:
        raise ValueError("parameter 'prob' in '%s.entropy' should sum to 1" % __name__)

    if data.any():
        num_samples    = data.shape[0]
        if len(data.shape) == 1:
            num_dimensions = 1
        else:
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
                x = np.reshape(x, (x.shape[0], 1))
            D = squareform(pdist(x, 'euclidean'))
            D = D + np.max(D)*np.eye(D.shape[0])
            return np.min(D, axis=0)

        if data is None:
            raise ValueError('Nearest neighbors entropy requires original data')

        if len(data.shape) > 1:
            k = num_dimensions
        else:
            k = 1

        Ak  = (k*np.pi**(float(k)/float(2)))/gamma(float(k)/float(2)+1)
        rho = getrho(data)

        if units is 'bits':
            # 0.577215... is the Euler-Mascheroni constant (np.euler_gamma)
            return k*np.mean(np.log2(rho)) + np.log2(num_samples*Ak/k) + np.log2(np.exp(1))*np.euler_gamma
        elif units is 'nats':
            # 0.577215... is the Euler-Mascheroni constant (np.euler_gamma)
            return k*np.mean(np.log(rho)) + np.log(num_samples*Ak/k) + np.log(np.exp(1))*np.euler_gamma
        else:
            print('Units not recognized: {}'.format(units))


    elif method == 'gaussian':
        from numpy.linalg import det

        if data is None:
            raise ValueError('Nearest neighbors entropy requires original data')

        detCov = det(np.dot(data.transpose(), data)/num_samples)
        normalization = (2*np.pi*np.exp(1))**num_dimensions

        if detCov == 0:
            return -np.inf
        else:
            if units is 'bits':
                return 0.5*np.log2(normalization*detCov)
            elif units is 'nats':
                return 0.5*np.log(normalization*detCov)
            else:
                print('Units not recognized: {}'.format(units))

    elif method == 'bin':
        if prob is None and bins is None:
            raise ValueError('Either prob or bins must be specified.')

        if data is not None:
            prob = symbols_to_prob(data, bins=bins)

        if units is 'bits':
            # compute the log2 of the probability and change any -inf by 0s
            logProb = np.log2(prob)
            logProb[logProb == -np.inf] = 0
        elif units is 'nats':
            # compute the log2 of the probability and change any -inf by 0s
            logProb = np.log(prob)
            logProb[logProb == -np.inf] = 0
        else:
            print('Units not recognized: {}'.format(units))

        # return sum of product of logProb and prob
        # (not using np.dot here because prob, logprob are nd arrays)
        return -float(np.sum(prob * logProb))


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


def mi(x, y, bins_x=None, bins_y=None, bins_xy=None, method='nearest-neighbors', units='bits'):
    '''
    compute and return the mutual information between x and y

    inputs:
    -------
        x, y:       numpy arrays of shape samples x dimension
        method:     'nearest-neighbors', 'gaussian', or 'bin'
        units:      'bits' or 'nats'

    output:
    -------
        mi:     float

    Notes:
    ------
        if you are trying to mix several symbols together as in mi(x, (y0,y1,...)), try

        info[p] = _info.mi(x, info.combine_symbols(y0, y1, ...) )
    '''
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

    # wrapped in try bracket because x, y might have no .shape attribute
    try:
        # handling for 1d np arrays
        if len(x.shape) == 1:
            x = np.expand_dims(x, 1)
        if len(y.shape) == 1:
            y = np.expand_dims(y, 1)
    except:
        pass

    HX  = entropy(data=x, bins=bins_x, method=method, units=units)
    HY  = entropy(data=y, bins=bins_y, method=method, units=units)
    HXY = entropy(data=np.concatenate([x, y], axis=1), bins=bins_xy, method=method, units=units)

    return HX + HY - HXY


def cond_entropy(x, y, bins_y=None, bins_xy=None, method='nearest-neighbors', units='bits'):
    '''
    compute the conditional entropy H(X|Y).

    method:     'nearest-neighbors', 'gaussian', or 'bin'
                if 'bin' need to provide bins_y, and bins_xy
    units:      'bits' or 'nats'
    '''
    HXY = entropy(data=np.concatenate([x, y], axis=1), bins=bins_xy, method=method, units=units)
    HY  = entropy(data=y, bins=bins_y, method=method, units=units)

    return HXY - HY
