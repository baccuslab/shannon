from nose.tools import assert_equal, assert_true, assert_raises, assert_almost_equal
from shannon import discrete
from shannon import bottleneck
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
import pdb

def setup():
  print("SETUP!")

def teardown():
  print("TEAR DOWN!")

def test_basic():
  print("I RAN!")

def test_Distribution():
    '''
    Make a few distributions and test that method 'sample' generates samples
    from those distributions.
    '''
    dist_0 = bottleneck.Distribution(np.array([1,1,1])/3)
    N = 1E7
    samples = dist_0.sample(N)
    for i in range(3):
        assert_almost_equal(dist_0.prob[i], np.mean(samples==i), 3)


    dist_1 = bottleneck.Distribution(np.array([0, .5, .25, .25]))
    N = 1E7
    samples = dist_1.sample(N)
    for i in range(len(dist_1.prob)):
        assert_almost_equal(dist_1.prob[i], np.mean(samples==i),3)

def test_remove_symbol_from_dist():
    dist = bottleneck.Distribution(np.array([0, .5, .25, .25]))

    # remove first symbol, dist should be identical
    new_dist = bottleneck.remove_symbol_from_dist(dist, 0)
    
    assert(type(new_dist) == type(dist))        # thest only once that type is not changed

    assert_array_equal(dist.prob, new_dist.prob) 

    # remove other symbols
    new_dist = bottleneck.remove_symbol_from_dist(dist, 1)
    assert_array_equal(np.array([0,0,.5,.5]), new_dist.prob) 

    new_dist = bottleneck.remove_symbol_from_dist(dist, 2)
    assert_array_equal(np.array([0, 2.0/3, 0, 1.0/3]), new_dist.prob) 

    new_dist = bottleneck.remove_symbol_from_dist(dist, 3)
    assert_array_equal(np.array([0, 2.0/3, 1.0/3, 0]), new_dist.prob) 


def test_1():
    """
    Make a toy example where:
        x is uniformly distributed with 2
        y = x if (x in [0,1]) and y is randomint(0,4) if x in [2,3]

        x = 0 and 1 carry 2 bits of information about y
        x = 2, 3 carry zero information about y

        I expect the bottleneck to be [2, 2, 0, 0]
    """
    '''
    N=2
    m = 10000
    x = randint(0, 2**N, size=m)
    y = where(x<2, x, randint(0,2**N, size=m))

    diff = bottleneck.differentiate_mi(x,y)
    return x, y, diff
    '''
    pass
