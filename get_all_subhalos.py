import sys
import numpy
import h5py
import glob
import json
import requests
import pandas


list_of_halo_catalogues  = sorted( glob.glob('*.hdf5'), key = lambda x: int(x.split("099.")[1].split('.hdf5')[0] ))
par = h5py.File(list_of_halo_catalogues[0],'r')['Parameters']
h = par.attrs.items()[9][1]

print list_of_halo_catalogues
stellar_mass_in_halos = numpy.array([])
subhalo_position = numpy.array([]).reshape(-1, 3)
photometrics = numpy.array([])

for catalogue_name in list_of_halo_catalogues:
    with h5py.File(catalogue_name, 'r') as f:
        try:
            stellar_mass_in_this_catalogue = f['Subhalo']['SubhaloMassType'][:,4]
            position = f['Subhalo']['SubhaloPos'][:]
            photo = f['Subhalo']['SubhaloStellarPhotometrics'][:,5]
            print position.shape
            stellar_mass_in_halos = numpy.hstack((stellar_mass_in_halos, stellar_mass_in_this_catalogue))
            photometrics = numpy.hstack((photometrics, photo))
            subhalo_position = numpy.vstack((subhalo_position, position))
        except:
            pass

subhalo_data = numpy.column_stack([photometrics, stellar_mass_in_halos, subhalo_position])
data = pandas.DataFrame({"r": subhalo_data[:,0],
                         "m": subhalo_data[:,1]*(1e10/h),
                         "x": subhalo_data[:,2]/1e3/h,
                         "y": subhalo_data[:,3]/1e3/h,
                         "z": subhalo_data[:,4]/1e3/h})
data.to_csv("illustristng_subhalos.csv", index=False)
