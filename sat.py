"""
Water vapor saturation functions
"""
import numpy as np

epsilon = 0.62197

def esatw(T):
    """
    Saturation vapor pressure over liquid water 
    Taken from ECMWF, which is based on Tetens formula: https://confluence.ecmwf.int/display/METV/Thermodynamic+functions
    Args:
     - T : kelvin
    Returns:
     - sat vapor pressure over liquid : Pa
    """
    es0 = 611.21  # saturation vapor pressure at T0 (Pa)
    T0  = 273.16  # (K)
    Ti  = T0 - 23 # (K)
    a3  = 17.502  # liquid water (Buck 1981)
    a4  = 32.19   # (K)
    return es0 * np.exp(a3 * (T - T0)/(T - a4))


def esati(T):
    """
    Saturation vapor pressure over ice 
    Taken from ECMWF, which says it is based on Tetens formula: https://confluence.ecmwf.int/display/METV/Thermodynamic+functions
    Args:
     - T : kelvin
    Returns:
     - sat vapor pressure over ice : Pa
    """
    es0 = 611.21  # saturation vapor pressure at T0 (Pa)
    T0  = 273.16  # (K)
    a3  = 22.587  # liquid water (Buck 1981)
    a4  = -0.7   # (K)
    return es0 * np.exp(a3 * (T - T0)/(T - a4))

    
def esat(T):
    """
    Generalized saturation vapor pressure function that returns value for liquid, ice, or in between
    depending on the input temperature. Uses linear interpolation between 250.16 and 273.16.
    Args:
     - T: kelvin
    Returns:
     - sat vapor pressure : Pa
    """
    T0 = 273.16  # (K)
    Ti = T0 - 23.0 # (K)

    liq_wgt = np.clip((T - Ti) / (T0 - Ti), a_min=0, a_max=1)
    return liq_wgt * esatw(T) + (1 - liq_wgt) * esati(T)


def wsat(T, p):
    """
    Generalized saturation mixing ratio.
    Linearly interpolates between ice and liquid saturation between 253.16 and 273.16 K.
    Args:
     - T: kelvin
     - p: Pa
    Returns:
     - sat mixing ratio in kg/kg
    """
    return epsilon * esat(T) / (p - esat(T))

def qsat(T, p):
    """
    Generalized saturation specific humidity.
    Linearly interpolates between ice and liquid saturation between 253.16 and 273.16 K.
    Args:
     - T: kelvin
     - p: Pa
    Returns:
     - sat mixing ratio in kg/kg
    """
    return epsilon * esat(T) / (p + (esat(T) * (epsilon - 1)))
