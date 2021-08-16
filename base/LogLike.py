import theano.tensor as tt
import numpy as np

# define a theano Op for our likelihood function
class LogLike(tt.Op):
    itypes = [tt.dvector]  
    otypes = [tt.dscalar]  

    def __init__(self, loglike, y0, tVector, data, sigma, allParams):
        # add inputs as class attributes
        self.likelihood = loglike
        self.y0 = y0
        self.tVector = tVector
        self.data = data
        self.sigma = sigma
        self.allParams = allParams

    def perform(self, node, inputs, outputs):
        # the method that is used when calling the Op
        (theta,) = inputs  # this will contain my variables

        # call the log-likelihood function
        logl = self.likelihood(theta, self.y0, self.tVector, self.data, self.sigma, self.allParams)

        outputs[0][0] = np.array(logl)  # output the log-likelihood