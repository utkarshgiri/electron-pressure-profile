import os
import numpy
import h5py
import glob
import pickle
import sys
import time
import logging
import particle_pressure
import msgpack
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
numpy.set_printoptions(threshold=numpy.nan)
from mpi4py import MPI
comm = MPI.COMM_WORLD
size = comm.size

halo_centers_filename = "./illustristng_halocenters_between_10_11.msg"

centers = numpy.load("center.npy")

allfiles = sorted(glob.glob("./snapshots/halo*.h5"), key=lambda x: int(x.split(".h5")[0].split("halo")[-1]))

h = 0.6774
skipby = 8
denom = 1000.*h
numerator = 1e10/h
for loop in range(0, len(allfiles), skipby):
    
    starting_file = loop
    ending_file = starting_file + skipby
    files = allfiles[starting_file:ending_file]

    total_files = len(files)
    files_this_core = numpy.array_split(numpy.array(range(total_files)), size)[comm.rank].tolist()
    files_this_core = numpy.array(files)[files_this_core]

    (radius, box_size, bin_size, bin_start, bin_end) = (10., 75./h, 20, -2, 1)
    cumulative_profile = numpy.zeros(shape=(bin_size))
    bins = numpy.logspace(bin_start,bin_end,bin_size+1)
    arr = numpy.array([]).reshape(0,8)
    for name in files_this_core:
        gas = h5py.File(name, 'r')
        arr = numpy.vstack((arr, numpy.column_stack([gas['PartType0']['Coordinates'][:,0]/denom,
                                                     gas['PartType0']['Coordinates'][:,1]/denom,
                                                     gas['PartType0']['Coordinates'][:,2]/denom,
                                                     gas['PartType0']['Masses'][:]*numerator,
                                                     gas['PartType0']['ElectronAbundance'][:],
                                                     gas['PartType0']['InternalEnergy'][:],
                                                     gas['PartType0']['StarFormationRate'][:],
                                                     gas['PartType0']['Density'][:]*numerator*denom])))
    comm.Barrier()
    
    for run in range(int(len(centers) / comm.size) ):

        x, y, z, m, a, e, d, v = [numpy.array([]) for _ in range(8)]
        sendsize, senddisp = [], [0]

        for i, center in enumerate(centers[run*comm.size:(run+1)*comm.size]):
            center_of_halo = [center['x'], center['y'], center['z']]
            suffix = run * comm.size + comm.rank
            xl = center['x'] - radius
            yl = center['y'] - radius
            zl = center['z'] - radius
            xh = center['x'] + radius
            yh = center['y'] + radius
            zh = center['z'] + radius

            region = numpy.copy(arr)

            if xl < 0 or xh > box_size:
                xln = xl % box_size
                xhn = xh % box_size
                region = region[numpy.logical_or(region[:,0] < xhn, region[:,0] > xln)]
                if xl < 0:
                    region[region[:,0] > (box_size - radius),0] -= box_size
                elif xh > box_size:
                    region[region[:,0] < radius,0] += box_size

            else:
                region = region[(region[:,0] < xh) & (region[:,0] > xl)]

            if yl < 0 or yh > box_size:
                yln = yl % box_size
                yhn = yh % box_size
                region = region[numpy.logical_or(region[:,1] < yhn, region[:,1] > yln)]
                if yl < 0:
                    region[region[:,1] > (box_size - radius), 1] -= box_size
                elif yh > box_size:
                    region[region[:,1] < radius, 1] += box_size

            else:
                region = region[(region[:,1] < yh) & (region[:,1] > yl)]

            if zl < 0 or zh > box_size:
                zln = zl % box_size
                zhn = zh % box_size
                region = region[numpy.logical_or(region[:,2] < zhn, region[:,2] > zln)]

                if zl < 0:
                    region[region[:,2] > (box_size - radius), 2] -= box_size
                elif zh > box_size:
                    region[region[:,2] < radius, 2] += box_size

            else:
                region = region[(region[:,2] < zh) & (region[:,2] > zl)]

            x = numpy.hstack((x, region[:,0]))
            y = numpy.hstack((y, region[:,1]))
            z = numpy.hstack((z, region[:,2]))
            m = numpy.hstack((m, region[:,3]))
            a = numpy.hstack((a, region[:,4]))
            e = numpy.hstack((e, region[:,5]))
            d = numpy.hstack((d, region[:,7]))
            v = numpy.hstack((v, region[:,3]/region[:,7]))

            sendsize.append(len(region[:,0]))
            senddisp.append(len(x))

        senddisp = senddisp[:-1]
        recvsize = comm.alltoall(sendsize)
        recvsize = [numpy.int64(xx) for xx in recvsize]
        recvdisp = [numpy.int64(0)]

        for i in range(comm.size-1):
            recvdisp.append(recvdisp[-1] + recvsize[i])

        xx = numpy.empty(numpy.sum(recvsize), dtype=numpy.float64)
        sendbuf = [ x, sendsize, senddisp, MPI.DOUBLE ]
        recvbuf = [ xx, recvsize, recvdisp, MPI.DOUBLE ]
        comm.Alltoallv(sendbuf, recvbuf)
        comm.Barrier()
        del x

        yy = numpy.empty(numpy.sum(recvsize), dtype=numpy.float64)
        sendbuf = [ y, sendsize, senddisp, MPI.DOUBLE ]
        recvbuf = [ yy, recvsize, recvdisp, MPI.DOUBLE ]
        comm.Alltoallv(sendbuf, recvbuf)
        comm.Barrier()
        del y

        zz = numpy.empty(numpy.sum(recvsize), dtype=numpy.float64)
        sendbuf = [ z, sendsize, senddisp, MPI.DOUBLE ]
        recvbuf = [ zz, recvsize, recvdisp, MPI.DOUBLE ]
        comm.Alltoallv(sendbuf, recvbuf)
        comm.Barrier()
        del z

        mm = numpy.empty(numpy.sum(recvsize), dtype=numpy.float64)
        sendbuf = [ m, sendsize, senddisp, MPI.DOUBLE ]
        recvbuf = [ mm, recvsize, recvdisp, MPI.DOUBLE ]
        comm.Alltoallv(sendbuf, recvbuf)
        comm.Barrier()
        del m

        aa = numpy.empty(numpy.sum(recvsize), dtype=numpy.float64)
        sendbuf = [ a, sendsize, senddisp, MPI.DOUBLE ]
        recvbuf = [ aa, recvsize, recvdisp, MPI.DOUBLE ]
        comm.Alltoallv(sendbuf, recvbuf)
        comm.Barrier()
        del a

        ee = numpy.empty(numpy.sum(recvsize), dtype=numpy.float64)
        sendbuf = [ e, sendsize, senddisp, MPI.DOUBLE ]
        recvbuf = [ ee, recvsize, recvdisp, MPI.DOUBLE ]
        comm.Alltoallv(sendbuf, recvbuf)
        comm.Barrier()
        del e

        dd = numpy.empty(numpy.sum(recvsize), dtype=numpy.float64)
        sendbuf = [ d, sendsize, senddisp, MPI.DOUBLE ]
        recvbuf = [ dd, recvsize, recvdisp, MPI.DOUBLE ]
        comm.Alltoallv(sendbuf, recvbuf)
        comm.Barrier()
        del d

        vv = numpy.empty(numpy.sum(recvsize), dtype=numpy.float64)
        sendbuf = [ v, sendsize, senddisp, MPI.DOUBLE ]
        recvbuf = [ vv, recvsize, recvdisp, MPI.DOUBLE ]
        comm.Alltoallv(sendbuf, recvbuf)
        comm.Barrier()
        del v

        number_of_gas_particles = len(dd)
        pressure, temperature = particle_pressure.compute(ee, aa, dd)
        del ee; del dd;
        gas_position = numpy.column_stack((xx, yy, zz))
        assert number_of_gas_particles == pressure.size, "pressure_n_temperature size is not equal  num_particle!"
        assert number_of_gas_particles == gas_position.shape[0], "gas_position size is not equal  num_particle!"


        center_of_halo = [centers[run*comm.size + comm.rank]['x'], centers[run*comm.size + comm.rank]['y'], centers[run*comm.size + comm.rank]['z']]
        pressure_in_bin = numpy.zeros(bin_size)
        density_in_bin = numpy.zeros(bin_size)
        distance = numpy.linalg.norm(gas_position - center_of_halo,axis=1)

        maximum = numpy.max(distance) if len(distance) > 0 else 0
        minimum = numpy.min(distance) if len(distance) > 0 else 0


        for i in range(bin_size):
            bin_volume = 4/3. * numpy.pi * (bins[i+1]**3 - bins[i]**3)
            cell_volume = vv[(distance > bins[i]) & (distance < bins[i+1])]
            cell_pressure = pressure[(distance > bins[i]) & (distance < bins[i+1])]
            cell_mass = mm[(distance > bins[i]) & (distance < bins[i+1])]
            #abundance = aa[(distance > bins[i]) & (distance < bins[i+1])]
            cell_temperature = temperature[(distance > bins[i]) & (distance < bins[i+1])]
            bin_mass = numpy.sum(cell_mass)
            volume_weighted_pressure = numpy.sum(cell_mass *  cell_temperature)
            pressure_in_bin[i] = volume_weighted_pressure / bin_volume
            density_in_bin[i] = bin_mass / bin_volume 
            """
            try:
                aa = numpy.load("mass_and_weighted_temp{}_{},{}_bin_{}.npy".format(center_of_halo[0], center_of_halo[1], center_of_halo[2], i))
                aa = aa + numpy.array([bin_mass, volume_weighted_pressure])
                numpy.save("mass_and_weighted_temp{}_{},{}_bin_{}.npy".format(center_of_halo[0], center_of_halo[1], center_of_halo[2], i), aa)
            except IOError:
                aa = numpy.array([bin_mass, volume_weighted_pressure])
                #print aa
                numpy.save("mass_and_weighted_temp{}_{},{}_bin_{}.npy".format(center_of_halo[0], center_of_halo[1], center_of_halo[2], i), aa)
            """
        if loop != 0:
            load_pressure = numpy.load("pressure_bw_10_11_%s.npy"%(run*comm.size + comm.rank))
            load_pressure = load_pressure + pressure_in_bin
            
            load_density = numpy.load("density_bw_10_11_%s.npy"%(run*comm.size + comm.rank))
            load_density = load_density + density_in_bin


        else:
            load_pressure = pressure_in_bin
            load_density = density_in_bin

        numpy.save("pressure_bw_10_11_%s"%(run*comm.size + comm.rank), load_pressure)
        numpy.save("density_bw_10_11_%s"%(run*comm.size + comm.rank), load_density)
comm.Barrier()
if comm.rank == 0:
    files = glob.glob("pressure_bw*.npy")
    pressure = numpy.zeros_like(numpy.load(files[0]))
    for name in files:
        pressure += numpy.load(name)
        os.remove(name)
    pressure = pressure / len(files)
    numpy.save(halo_centers_filename.split(".msg")[0], numpy.column_stack(((bins[:-1] + bins[1:])/2., pressure)))
    solar_mass_to_grams = 2e33
    mpc_to_cm = 3.086e24
    k = 8.617e-8
    mu = 0.59
    hydrogen_mass = 1.67e-24
    factor = ( solar_mass_to_grams * k )/ ( mu * hydrogen_mass * mpc_to_cm**3)
    plt.loglog((bins[:-1] + bins[1:]) / 2., pressure*factor)
    plt.xlabel(r'$\mathrm{r \ [Mpc]}$')
    plt.ylabel(r'$\mathrm{P \ [Kev/cm^{3}]}$')
    plt.savefig(halo_centers_filename.split(".msg")[0] + ".pdf")




