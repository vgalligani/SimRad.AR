# Python script to _________________________
# V. Galligani 
# CIMA, UBA-CONICET, Argentina

from pytmatrix.psd import PSD, np
try:
    import cPickle as pickle
except ImportError:
    import pickle
import warnings

class NEWhybridPSD(PSD):
    """Hybrid PSD for WRF WDM6 cloud (PSD).

    Callable class to provide an exponential PSD with the given
    parameters. The attributes can also be given as arguments to the
    constructor.

   The PSD form is:
    N(D) = N0 * D**2 exp(-(Lambda*D)**3)

    Attributes:
        N0: the intercept parameter.
        Lambda: the inverse scale parameter
        D_max: the maximum diameter to consider (defaults to 11/Lambda,
            i.e. approx. 3*D0, if None)

     Args (call):
        D: the particle diameter.

    Returns (call):
        The PSD value for the given diameter.
        Returns 0 for all diameters larger than D_max.

   """

    def __init__(self, N0=1.0, Lambda=1.0, D_max=None):
        self.N0 = float(N0)
        self.Lambda = float(Lambda)
        self.D_max = 11.0/Lambda if D_max is None else D_max

    def __call__(self, D):
        # For large mu, this is better numerically than multiplying by D**mu
        psd = self.N0 * np.exp(  (2.*np.log(D))-((self.Lambda*D)**3))
        if np.shape(D) == ():
            if D > self.D_max:
                return 0.0
        else:
            psd[D > self.D_max] = 0.0
        return psd

    def __eq__(self, other):
        try:
            return isinstance(other, NEWhybridPSD) and \
                (self.N0 == other.N0) and (self.Lambda == other.Lambda) and \
                (self.D_max == other.D_max)
        except AttributeError:
            return False




class CHECKExponentialPSD(PSD):
    """Exponential particle size distribution (PSD).
    
    Callable class to provide an exponential PSD with the given 
    parameters. The attributes can also be given as arguments to the 
    constructor.

    The PSD form is:
    N(D) = N0 * exp(-Lambda*D)

    Attributes:
        N0: the intercept parameter.
        Lambda: the inverse scale parameter        
        D_max: the maximum diameter to consider (defaults to 11/Lambda,
            i.e. approx. 3*D0, if None)

    Args (call):
        D: the particle diameter.

    Returns (call):
        The PSD value for the given diameter.    
        Returns 0 for all diameters larger than D_max.
    """

    def __init__(self, N0=1.0, Lambda=1.0, D_max=None):
        self.N0 = float(N0)
        self.Lambda = float(Lambda)
        self.D_max = 11.0/Lambda if D_max is None else D_max

    def __call__(self, D):
        psd = self.N0 * np.exp(-self.Lambda*D)
        if np.shape(D) == ():
            if D > self.D_max:
                return 0.0
        else:
            psd[D > self.D_max] = 0.0
        return psd

    def __eq__(self, other):
        try:
            return isinstance(other, CHECKExponentialPSD) and \
                (self.N0 == other.N0) and (self.Lambda == other.Lambda) and \
                (self.D_max == other.D_max)
        except AttributeError:
            return False


