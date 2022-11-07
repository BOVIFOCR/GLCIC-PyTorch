import csv
import sys

fname = sys.argv[1]

dct = {}

with open(fname, 'r') as fd:
	ls = csv.reader(fd)
	
	for l in ls:
		if l[5] not in dct.keys():
			dct[l[5]] = []
		dct[l[5]].append(",".join([str(int(x)) for x in l[1:5]]))

for k in dct.keys():
	with open("labels/" + k + ".txt", 'w') as fd:
		for x in dct[k]:		
			fd.write(x + "\n")
