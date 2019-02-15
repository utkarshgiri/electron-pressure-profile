import numpy
import pandas
import msgpack
import copy
import h5py


par = h5py.File("./fof_subhalo_tab_099.0.hdf5",'r')['Parameters']
h = par.attrs.items()[9][1]

subhalos = pandas.read_csv("illustristng_subhalos.csv", header=0)
subhalos = subhalos[subhalos['m'] > 1e7]

subhalos['more_luminous'] = numpy.zeros(len(subhalos))
subhalos['r'] = numpy.abs(subhalos['r'])
cutoff_for_neighbour = 1.0

mass_range = numpy.linspace(10, 12.0, 21)

for mass in range(len(mass_range)-1):
    stellar_mass_above = 10**mass_range[mass]
    stellar_mass_below = 10**mass_range[mass+1]

    selected = copy.deepcopy(subhalos[subhalos['m'] > stellar_mass_above])
    selected = copy.deepcopy(selected[selected['m'] < stellar_mass_below])


    for i, subhalo in selected.iterrows():
        
        neighbours = copy.deepcopy(subhalos)
        neighbours['distance'] = numpy.zeros(len(neighbours))
        neighbours['distance'] = numpy.linalg.norm(neighbours.iloc[:,2:5].values - subhalo[2:5].values, axis=1)
        neighbours = neighbours[neighbours['distance'] < cutoff_for_neighbour]
        neighbours = neighbours[neighbours['r'] > subhalo['r']]
        if len(neighbours) > 0:
            selected.loc[i, 'more_luminous'] = 1
        else:
            print "good"
       
    halos_of_worth = selected[selected['more_luminous'] == 0]
    halos_of_worth.to_csv("illustris_halos_gt_{}_lt_{}.csv".format(mass_range[mass], mass_range[mass+1]), index=False)
    halolist = []
    for ii, item in halos_of_worth.iterrows():
        halolist.append({'x': item['x'],
                        'y': item['y'],
                        'z': item['z']})

    with open("illustris_halos_gt_{}_lt_{}.msg".format(mass_range[mass],mass_range[mass+1]), 'w') as outfile:
        msgpack.pack(halolist, outfile)

