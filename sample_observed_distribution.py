import numpy
import pandas
import msgpack
import glob

msgfiles = sorted(glob.glob("./illustris_halos_gt_10.?_lt_1?.?.msg") + glob.glob("./illustris_halos_gt_11.?_lt_1?.?.msg"))

numbers_sim = []
for name in msgfiles:
    numbers_sim.append(len(msgpack.load(open(name,'r'))))

bins_sim = []
for name in msgfiles:
    bins_sim.append(float(name.split("gt_")[-1].split("_lt")[0]))

numbers_observed = numpy.array([3847,5224,7499,9862, 13220,17328, 21620, 25309, 27866, 28325, 26026, 22085, 16871,11615, 7160,
                       3664,1624,573,145, 44])

bins_obs = numpy.logspace(10, 12, len(numbers_observed)+1)

f = [] 
for i in range(len(bins_sim)):
    f.append((numbers_observed[i] * 4*256.)/ (numbers_sim[i] * numpy.sum(numbers_observed)))

number_keep = []
for i in range(len(bins_sim)):
    number_keep.append(f[i] * numbers_sim[i])

sample = numpy.around(number_keep)

sims = []
for i, name in enumerate(msgfiles[:10]):
    sims.append(numpy.random.choice(numpy.array(msgpack.load(open(name,'r'))), int(sample[i])))

flat_list = [item for sublist in sims for item in sublist]

print len(flat_list)

with open("illustristng_halocenters_between_10_11.msg", "w") as h:
    msgpack.pack(flat_list, h, encoding='utf-8')
