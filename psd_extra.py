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


class ThomPSD(PSD):
    """Hybrid PSD for WRF THOM snow (PSD).

    Callable class to provide a PSD with the given
    parameters. The attributes can also be given as arguments to the
    constructor.

   The PSD form is:
    N(D) = N1 exp(-(Lambda1*D) + N2 D**mu exp-(Lambda2*D)

     Args (call):
        D: the particle diameter.

    Returns (call):
        The PSD value for the given diameter.
        Returns 0 for all diameters larger than D_max.

   """

    def __init__(self, N1=1.0, N2=1.0, Lambda1=1.0, Lambda2=1.0, mu=0.0, D_max=None):
        self.N1 = float(N1)
        self.N2 = float(N2)
        self.Lambda1 = float(Lambda1)
        self.Lambda2 = float(Lambda2)
        self.mu = float(mu)
        self.D_max = 11.0/Lambda if D_max is None else D_max
        
    def __call__(self, D):
        # For large mu, this is better numerically than multiplying by D**mu
        psd2 = self.N2 * np.exp(self.mu*np.log(D)-self.Lambda2*D)
        psd1 = self.N1 * np.exp(-self.Lambda1*D)
        psd  = psd1 + psd2 

        if np.shape(D) == ():
            if D > self.D_max:
                return 0.0
        else:
            psd[D > self.D_max] = 0.0
        return psd

    def __eq__(self, other):
        try:
            return isinstance(other, ThomPSD) and \
                (self.N1 == other.N1) and (self.N2 == other.N2) and \
                (self.Lambda1 == other.Lambda1) and (self.Lambda2 == other.Lambda2) and \
                (self.mu == other.mu) and (self.D_max == other.D_max)
        except AttributeError:
            return False
