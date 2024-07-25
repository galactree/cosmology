import numpy as np
import scipy
from cosmology.utilities import *

class Component:
    '''
    
    companion object for Cosmology components
    
    ###########
    Parameters

    O_0, float: initial density relative to critical density


    name, str: name of the component


    w, float or callable: value of omega in the equation of state,
                        if w is a float: equation of state is constant with redshift
                        if w is callable: equation of state varies with redshift and omega is a function of z.
    
    
    '''
    def __init__(self, O_0, w, name):
        self.O_0 = O_0 # initial density relative to critical density
        self.name = name # name of component
        self.w = w # equation of state, omega
        self.w_func = callable(w) # if omega is a function of z
        # flag for if numerical integration required
        

    def density_function(self, z):
        '''

        Compute the energy density for component within a cosmology

        ###########
        Parameters
        
        z, float: the redshift to compute the value of the component energy density at z

        ########
        Returns

        Energy density of component at z

        '''


        if self.w_func: # if omega is a function of z
            integrand = lambda x: 3*(1+self.w(x)) # convert w(z) into the integrand
            I_z, error = scipy.integrate.quad(integrand, 0,z) # unpack integral result and error
            density = self.O_0*np.exp(I_z)# evaluate solution,

            return density
        
        else: # if omega is constant
            alpha = 3*(1+self.w)
            return self.O_0*(1+z)**(alpha)
