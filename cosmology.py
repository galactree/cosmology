import numpy as np
import scipy.integrate
from cosmology.component import Component
from cosmology.utilities import *

class Cosmology:
    '''


    For computing cosmological parameters for given components


    ############
    Parameters
    H0, float: the Hubble constant today (km/s/Mpc)
    Om0, float (default=0.3): initial matter density relative to the critical density 
    Ode0, float (default=0.7): initial dark energy density relative to the critical density

    ############
    kwargs:
    Orad0, float: initial radiation energy density relative to the critical density (default, 0)
    Ok0, float: initial spatial curvature relative to the critical density (default, 0)
    w_m, w_de, w_rad, w_k, float or callable: equation of state for each component
    Ob0 (float,): baryon density units, h^2
    '''
    def __init__(self, H0, Om0=0.3, Ode0=0.7, **kwargs):
        self.c = 2.99792e5 # speed of light km/s
        self.G = 4.301e-9 # Gravitational constant in km^2 Mpc/ M_sun/s^2
        self.H0 = 100/self.c # hubble constant in h Mpc^-1
        self.h = H0/100 # h value
        self.H_inv = self.c/100 # 1/H0 in h^-1 Mpc
        
        # components
        self.m = None # matter component
        self.de = None # dark energy component
        self.rad = None # radiation component
        self.k = None # curvature component
        self.create_component(Om0, Ode0, kwargs) # create components as prescribed by inputs
        self.components = []
        

        # store baryon density if provided
        if "Ob0" in kwargs.keys():
            self.Ob0 = kwargs['Ob0']/self.h**2 # units h^2
        

        # otherwise use fiducial value
        else:
            self.Ob0 = 0.0224/self.h**2 # units h^2


        # store non-zero components in component list
        for c in [self.m, self.de, self.rad, self.k]:
            if c is not None:
                self.components.append(c)
        
        # compute the initial critical density
        self.critical_density0 = self.critical_density(0) # critical density at z=0 in units h^-1 M_sun/ (h^-1 Mpc)^3
        

        # initialise ode solution approximations for linear growth suppression factor using default domain and ICs
        self.g_func = self.linear_growth_suppression_factor_solver()

        # initialise ode solution approximations for comoving distance using domain 0<=z<=10 and IC r(0)=0
        self.fast_r = self.r_ode(10)



        self.colours = ["#D81B60", "#FFC107", "#1E88E5", "#004D40"]

        # value of g(z=0)
        self.g0 = self.linear_growth_suppression_factor(0)
    
    def create_component(self, Om0, Ode0, kwargs):
        ''' 

        Create component objects to track the energy densities

        ################
        Parameters

        Om0 (float, between 0 and 1): matter energy density relative to critical (required)
        
        Ode0 (float, between 0 and 1): dark energy density relative to critical (required)

        ###############
        Returns

        None

        ###############
        kwargs:

        
        '''

        # component names
        component_names = {'m':'matter', 'de': 'dark_energy', 'rad': 'radiation','k':'curvature'}

        # default EoS
        defaults_w = {'m':0,'de':-1,'rad': 1/3, 'k': -1/3}

        # keep total of Omega ensure it =1
        O_tot = 0

        # check kwargs for radiation component parameters, if not provided use the defaults
        if 'Orad0' in kwargs.keys():
            O_tot += kwargs['Orad0']
            if 'w_rad' in kwargs.keys():
                    self.rad = Component(kwargs['Orad0'], kwargs[f'w_rad'], component_names['rad'])

            else:
                self.rad = Component(kwargs['Orad0'], defaults_w['rad'], component_names['rad'])
        
        # check kwargs for curvature component parameters, if not provided use the defaults
        if 'Ok0' in kwargs.keys():
            O_tot += kwargs['Ok0']

            if 'w_k' in kwargs.keys():
                self.k = Component(kwargs['Ok0'], kwargs[f'w_k'], component_names['k'])

            else:
                self.k = Component(kwargs['Ok0'], defaults_w['k'], component_names['k'])

        
        # check that total density doesnt not exceed 1 if 

        assert (O_tot+Om0+Ode0 )==1, 'total energy density of components relative to critical density does not =1'

        # check for matter component parameters, if not provided use defaults
        if Om0!=0:
            if "w_m" in kwargs.keys():
                self.m = Component(Om0, kwargs['w_m'], component_names['m'])
            else:
                self.m = Component(Om0, defaults_w['m'], component_names['m'])

        # check for dark energy component parameters, if not provided use defaults
        if Ode0!=0:
            if "w_de" in kwargs.keys():
                self.de = Component(Ode0, kwargs['w_de'], component_names['de'])
            else:
                self.de = Component(Ode0, defaults_w['de'], component_names['de'])


    

    def critical_density(self, z):
        '''
        
        find the critical density at the given z in natural units (h^-1 M_sun/ (h^-1 Mpc)^3)
        
        #########
        Parameters:
        z, float: Redshift to compute the critical density
        
        #########
        Returns:
        critical density in units h^-1 M_sun/ (h^-1 Mpc)^3
        
        '''


        return 3*((100*self.E(z))**2)/(8*np.pi*self.G)
    
        
    def E(self, z):
        '''

        Dimensionless RHS of Friedmann 1
        
        ##########
        Parameters
        z, float: Redshift at which to compute the Hubble parameter


        ##########
        Returns

        Dimensionless E(z)
        '''

        # Number of components
        N = len(self.components)
        res = np.zeros(N)


        # calculate each components density
        for i, c in enumerate(self.components):
            if self.components[i].w_func:
                    res[i] = c.density_function(z)
            else:
                res[i] = c.density_function(z)
        
        # add up the component energy densities
        return np.sqrt(res.sum())
    

            
    
    def r(self, z):
        """
        Comoving distance from z=0 to given redshift in h^-1 Mpc, using quadrature method of integration
        

        #########
        Parameters
        z, float: Redshift for which to compute the comoving distance

        
        #########
        Returns

        r (float, ): Comoving distance in h^-1 Mpc
        
        """

        # r(z) is defined by an integral of 1/E(z') from 0 to z'=z
        integrand = lambda z_: 1/(self.E(z_))

        # compute integral
        r = scipy.integrate.quad(integrand, 0, z)[0]
        return r
    
    
    def r_ode(self, max_z):
        """
        
        Calculate r(z) over an interval of z values which must include z=0 using the ODE method (FLRW)
        
        #########
        Parameters

        max_z (float, ): the redshift value to include in the domain of the output approx.

        #########
        Returns
        approximation to solution to r(z) ODE in h^-1 Mpc
        
        """
        # computes the RHS of the IVP
        r_ivp = lambda z, r: 1/self.E(z)

        # solve the IVP
        result = scipy.integrate.solve_ivp(r_ivp, [0, max_z], [0], dense_output=True)

        # return the approximation function in h^-1 Mpc
        return result.sol
    
    def faster_r(self, z):
        '''
        
        Compute the comoving distance, r(z) in h^-1 Mpc, using an approximated solution to the solution which is valid for 0<=z<=10


        #########
        Parameters

        z (listlike or float,) : redshift to compute the comoving distance 

        #########
        Returns

        r (listlike or float,): comoving distance in h^-1 Mpc corresponding to the input redshift.
        '''

        # use the approximation function that's computed upon initialisation

        r = self.fast_r(z)

        return r

    def dimensionless_growth_ode(self, a, x):
        '''

        Method which describes the RHS of the dimensionless growth ODE (Equation 1)

        #########
        Parameters

        a (float) : scale factor

        x (listlike, ): the vector on the RHS of the system of ODES, [dg/da, g]

        #########
        Returns

        X (listlike, ): the LHS of the system of ODEs [d^g/da^2, dg/da]

        '''

        # make sure the input vector is an array
        x = np.array(x)

        # convert a values to z to compute other parameters
        z = (1/a)-1

        # Dark energy denisty of z
        if self.de is not None:
            O_de = ((self.de).density_function(z))/self.E(z)**2
                    # check whether w depends on z and compute appropriately
            if self.de.w_func:
                w_de = (self.de).w(z)
            else:
                w_de = self.de.w
        else:
            O_de = 0
            w_de = -1
        

        # # check whether w depends on z and compute appropriately
        # if self.de.w_func:
        #     w_de = (self.de).w(z)
        # else:
        #     w_de = self.de.w

        
        # Matrix defined in equation (3)
        coeff = np.array([[(-1/(2*a))*(7-3*w_de*O_de), (-3/(2*(a**2)))*(1-w_de)*O_de], [1,0]])


        # compute matrix multiplication
        X = np.matmul(coeff, x)
        return X
    
    def linear_growth_suppression_factor_solver(self, range = [1/6,1],  ICs=[0,1], N=10000, return_approx=True):
        '''
        
        Solve the linear growth ODE for the supression factor, g.

        #########
        Parameters

        range (list, len 2): values of the scale factor, a, which define the endpoints of the domain over which to solve the ODE

        ICs (list, len 2): the initial condition to impose, must correspond to the value of dg/da and g at the lowest value of a in the provided domain.
                            the first entry must correspond to the value of dg/da
                            the second to the value of g

        N (int, opt) : the number of points in the domain to return values of g at

        return_approx (bool, opt): If True, an appoximate solution in the domain provided is returned.


        #########
        Returns

        solution (scipy.integrate.OdeSolution) : solution instance, if return_approx was True.

        g (arraylike, shape (N,)) : the g values corresponding to the a values passed, if return_approx was False

        t (arraylike, shape (N,)) : the a values in the domain corresponding to returned g values, if return_approx was False

        '''

        # solve IVP for ICs on input domain
        result = scipy.integrate.solve_ivp(self.dimensionless_growth_ode, range, ICs, t_eval = np.linspace(range[0], range[1], N), dense_output=True)

        # return approximation if required
        if return_approx:
            return result.sol
        
        # return g, a values otherwise
        else:
            return result.y[1], result.t

    
    def linear_growth_suppression_factor(self, z):
        '''
        
        Linear growth supression factor as a function of z,

        ############
        Parameters

        z (arraylike or float): redshift values to compute g

        ############
        Returns

        g (arraylike or float): g corresponding to input redshifts

        '''

        # convert z values to a
        a = 1/(1+z)

        # compute g(z) using initialised approximation
        return self.g_func(a)[1]
    
    def linear_growth_rate(self, z):
        '''
        
        Compute the value of the linear growth rate, D, at a given redshift(s)

        ##########
        Parameters

        z (arraylike or float, ): the redshift(s) to compute the linear growth rate


        ##########
        Returns

        D (arraylike or float, ): the linear growth rate at the input redshift,
                                    type matches the type of z inputted
        
        
        '''

        # compute g(z)
        g = self.linear_growth_suppression_factor(z)

        # g(z=0)
        g0 = self.g0
        

        # compute D(z)
        D = g/(g0*(1+z))

        return D
    
    def transfer_function(self, k):
        '''

        The linear transfer function, records the effect of the transition from radiation domination to matter on growth of density fluctuations (Bardeen et al. 1986)

        #########
        Parameters
        k (listlike, or float) : wavenumber k/(h Mpc^-1)

        ########
        Returns

        T(k) (listlike, or float) : the value(s) of the transfer function for input k



        '''

        # Definition of Gamma
        Gamma = ((self.m).O_0)*(self.h)*np.exp(-self.Ob0-(1.3*self.Ob0/(self.m).O_0))

        # Definition of q
        q = k/Gamma

        # Transfer Function
        T = (np.log(1+2.34*q)/(2.34*q)) * ((1 + 3.89*q +(16.1*q)**2 + (5.46* q)**3 + (6.71*q)**4)) ** (-0.25)


        return T
    

    def linear_power_spectrum(self, k, z, n_s=0.966, A_s = 2.1*1e-9, k_piv=0.05, use_approx=False):
        '''

        Linear power spectrum of matter density perturbations for FLRW cosmology (see Huterer 9.61).

        ##########
        Parameters

        k (listlike, or float) : wavenumber k/(h Mpc^-1)

        z (float): redshift


        
        n_s (float, optional): Scalar Spectral Index (default=0.966)
        A_s (float, optional):  Normalization of the power spectrum (default= 2.1*1e-9)
        k_piv (float, optional): "pivot" wavenumber, the wavenumber at whcih the promordial power is best constrained (default = 0.05 Mpc^-1, the Planck value)

        
        #########
        Returns

        Delta (listlike, or float): the value of the linear power spectrum of matter density perturbations for given wavenumber

    
        '''
        
        # prefactor with no z dependence
        prefactor = ((4*A_s/(25*((self.m).O_0)**2))*((k*self.h/k_piv)**(n_s-1))*((k*self.H_inv)**4))
        
        # calculate the transfer function and multiply by the prefactor term, this new term now contains all the k-dependence
        k_terms = prefactor*((self.transfer_function(k))**2)

        # use the linear growth supression factor method which is parameterised in z to find the appropriate g(z) value for input redshift
        # include the correct prefactor in z, 1/(1+z)
        g = (1/(1+z)) * self.linear_growth_suppression_factor(z)
        
        #square and combine according to eq 9.61 in Huterer,
        Delta = k_terms * g**2

        return Delta
    
    def amplitude_mass_fluct_0(self, M):
        '''

        Compute the RMS amplitude of the linear mass fluctuations

        ##########
        Parameters

        M (listlike or float): Mass of region, units solar masses

        #########
        Returns
        
        sigma (listlike or float, ): the amplitude of the mass fluctuations over smoothed of the region

        
        '''

        # calculate the energy density in units (h^-1 M_sun/ (h^-1 Mpc)^3)
        rho_m = self.m.O_0 * self.critical_density0

        # Mass to R (units, h^-1 Mpc)
        R = (3*M/(4*np.pi*rho_m))**(1/3)
        
        # define function to integrate over, corresponds to equation (9.84) in Huterer
        I = lambda x, r: (1/x)*self.linear_power_spectrum(x,0) * ((3*j_1(x*r)/(x*r))**2)

        # allow integration over different values of R
        amp = lambda R_: scipy.integrate.quad(I, 0, np.inf, args=(R_,))[0]

        # vectorise to use an array of R values
        vectorized_amp = np.vectorize(amp)

        # take the square root of the integral computed over an array of R
        sigma =  np.sqrt(vectorized_amp(R))

        return sigma
    
    def press_schecter_z(self, M, z=np.linspace(0,5,100), delta_c = 1.686):
        """

        Compute the Press Schecter Mass Function for the number density of objects in an interval dlnM around a mass M

        ###########
        Parameters
        
        M (float,): Mass in h^-1 M_sun

        z (listlike, )

        ###########
        Returns

        dn/dlnM (listlike,): PSMF of M of z

        """

        D = self.linear_growth_suppression_factor(z)/(0.78*(1+z))
        
        # convert to work in log M
        ln_sigma_0 = lambda m: np.log(self.amplitude_mass_fluct_0(np.exp(m)))

        # take the derivative
        derivative = np.abs(deriv_central_diff(np.log(M),ln_sigma_0, 0.001))

        # amplitude of mass fluctuations at z = 0
        sigma_0 = self.amplitude_mass_fluct_0(M)

        # amplitude of mass fluctuations for each z
        sigma = sigma_0 * D

        # combine to produce press-schecter mass function
        dn_dlnM = np.sqrt(2/np.pi) * (self.m).O_0* 2.775*1e11 * delta_c * derivative *np.exp(-delta_c**2/(2*(sigma)**2))/(M*sigma)

        return dn_dlnM

    def press_schecter(self, M = np.logspace(13,15,3), z=np.linspace(0,5,100), delta_c = 1.686):
        """
        
        Pass multiple M values to the Press Schecter Mass Function

        ###########
        Parameters
        
        M (listlike, ) : masses to compute dn/dlnM around

        z (listlike or float,) : redshift(s) to compute dn/dlnM at

        ############
        Returns

        dn/dlnM (listlike, ):  the number density of objects in an interval dlnM around a mass M

        """

        # for each M compute sigma^2(M,z)
        func_ = lambda m: self.press_schecter_z(m, z, delta_c) 
        return np.array([func_(m_) for m_ in M])
    

    
    def angular_diameter_distance(self, z):
        """
        Angular diameter distance between z=0 to given redshift in Mpc
        Parameters
        z, float: Redshift for which to compute the angular diameter distance

        Returns
        angular diameter distance in Mpc
        """
        r = self.faster_r(z)
        return r/(1+z)
    
    
    def luminosity_distance(self, z):
        """
        Luminosity distance between z=0 to given redshift in Mpc
        Parameters
        z, float: Redshift for which to compute the luminosity distance

        Returns
        Luminosity distance in Mpc
        """
        return (1+z)*self.faster_r(z)
    

    
    def comoving_V_element(self, z):
        """ Comoving volume element at a given redshift
        Parameters
        z, float: redshift for which to compute the comoving volume element
        Returns
        Comoving Volume Element in natural units
        """
        co_vec  = np.vectorize(lambda z_: (self.H_inv**3)*(self.fast_r(z_))**2/self.E(z_))
        return co_vec(z)
    
    
    # def age(self, z):
    #     """ Age of the Universe at a given redshift
    #     Parameters
    #     z, float: redshift for which to compute the age of the Universe

    #     Returns
    #     Age of the Universe in Gyr
    #     """
    #     integrand = lambda z_: (self.H(z_)*(1+z_))**(-1)
    #     integral = scipy.integrate.quad(integrand, z, np.inf)[0]
    #     age_in_s = self.converter.Mpc_to_km(integral)
    #     age_in_Gyr = self.converter.s_to_year(age_in_s)*self.converter.to_giga
    #     return age_in_Gyr
    



