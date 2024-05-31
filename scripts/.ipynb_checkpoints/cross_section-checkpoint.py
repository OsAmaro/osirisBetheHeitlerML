# original python script written by Bertrand Martinez @ GoLP-EPP (2022)

# numpy
import numpy as np

# scipy
from scipy.integrate import quad
from scipy.constants import c, e, m_e, epsilon_0, mu_0, N_A, k, h, hbar, alpha, value

r_e = value('classical electron radius')

def ltf(Z):
    '''
    This function determines the Thomas Fermi length for a given atomic number Z
    inputs :
        Z is th atomic number
    outputs :
        result is the TF length, normalised by the compton wavelength
    '''
    
    compton_wavelength = hbar / (m_e * c)
    length = (4. * np.pi * epsilon_0 * hbar ** 2 / (m_e * e ** 2) * Z ** (-1./3.))
    result = 0.885 * length / compton_wavelength

    return result

def I_1(d, l, q=1.0):
    
    '''
    This function computes the term I1 in the Bremsstrahlung and Bethe-Heitler cross-section
    inputs :
        d is the delta parameter
        l is the screening length
        q is a fraction between 0 and 1 for this screened potential
    outputs :
        result is the term I1
    '''

    T1 = l * d * ( np.arctan(l * d) - np.arctan(l) )
    T2 = - (l ** 2 / 2.) * (1. - d) ** 2 / (1. + l ** 2)
    T3 = (1. / 2.) * np.log((1. + l ** 2.) / (1. + (l * d) ** 2))

    result = q ** 2 * (T1 + T2 +T3)

    return result


def I_2(d, l, q=1.0):

    '''
    This function computes the term I2 in the Bremsstrahlung and Bethe-Heitler cross-section
    inputs :
        d is the delta parameter
        l is the screening length
        q is a fraction between 0 and 1 for this screenedpotential
    outputs :
        result is the term I2
    '''

    T1 = 4. * (l * d) ** 3 * (np.arctan(d * l) - np.arctan(l))
    T2 = (1. + 3. * (l * d) ** 2) * np.log((1. + l ** 2) / (1. + (l * d) ** 2))
    T3 = (6. * l ** 4 * d ** 2) * np.log(d) / (1. + l ** 2)
    T4 = l ** 2 * (d - 1.) * (d + 1. - 4. * l ** 2 * d ** 2) / (1. + l ** 2)

    result = 0.5 * q * (T1 + T2 + T3 + T4)
    
    return result

def bh_cs_dif(gp, k, Z):

    '''
    This function computes the differential Bethe-Heitler cross-section
    inputs :
        gp is the energy of the positron
        k is the energy of the photon
        Z is the atomic number
    outputs :
        result is the differential cross-section in m^2
    '''

    result = 0.0
    condition = (k >= 2.) and (gp >= 1.) and (gp <= k-1.)

    if condition :

        q  = 1.0
        ge = k - gp
        d  = k / (2.0 * gp * ge)
        l  = ltf(Z)

        T1 = 4. * (Z * r_e) ** 2 * alpha / k ** 3
        T2 = (gp ** 2 + ge ** 2) * (I_1(d, l, q) + 1.0)
        T3 = (2. / 3.) * gp * ge * (I_2(d, l, q) + 5. / 6.)
        
        result = T1 * (T2 + T3)
    
    return result


def bh_cs(Z, k):

    '''
    This function computes the total Bethe-Heitler cross-section
    inputs :
        Z is the atomic number
        k is the energy of the photon
    outputs :
        result is the total cross-section in m^2
    '''

    result = 0.0
    condition = (k > 2.0)

    if condition :
        result = quad(bh_cs_dif, 1.0, k-1.0, args=(k, Z))[0]

    return result

def bh_cdf(Z, k, gp):

    '''
    This function computes the CDF of the Bethe-Heitler cross-section
    inputs :
        Z is the atomic number
        k is the energy of the photon
        gp is the energy of the positron
    outputs :
        result is the CDF of the Bethe-Heitler cs (no units and between 0 and 1 by definition)
    '''

    condition = (k >= 2.) and (gp >= 0.) and (gp <= 1.)
    
    result = 0.0
    if condition :
        gp =  1.0 + (k - 2.) * gp
        numerator = quad(bh_cs_dif, 1.0, gp, args=(k, Z))[0]
        denominator = bh_cs(Z, k)
        result = numerator / denominator
    
    return result