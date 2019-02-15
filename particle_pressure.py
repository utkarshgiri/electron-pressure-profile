import numpy
import h5py

mp = 1.67e-24 # mass of proton in g
XH = 0.76     # hydrogen mass fraction

unit_energy_by_unit_mass = 1e10
kb = 1.38e-16

def compute(e, a, d):

    gamma  = 5./3.
    internal_energy = e #handle[:,5]
    electron_abundance = a #handle[:,4]
    gas_density = d / mp
    electron_density = electron_abundance * XH * gas_density

    denom = (1. + 3*XH + 4*XH*electron_abundance)

    mu = (4. * mp)/denom

    temperature = (gamma - 1.) * internal_energy * unit_energy_by_unit_mass * mu  / kb

    electron_pressure = electron_density*temperature

    return electron_pressure, temperature
    #return numpy.column_stack([electron_pressure, temperature])

