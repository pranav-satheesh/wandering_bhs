import h5py
import numpy as np

f = h5py.File("Testfile.h5", 'w')

testSnapDict = {"redshift": 3, 
                1: {"masses": [1, 2, 3], "rad": [4, 5, 6], "lum": [7, 8, 9], "total mass": 6, "stellar mass": 5, "halfrad": 15, "central": True}, 
                2: {"masses": [2, 3, 4], "rad": [5, 6, 7], "lum": [8, 9, 10], "total mass": 9, "stellar mass": 6, "halfrad": 10, "central": False}}

red = f.create_dataset("redshift", data=[3])
sub1 = f.create_group("sub1")
sub1masses = sub1.create_dataset("masses", data = [1, 2, 3])
sub1rad = sub1.create_dataset("rad", data = [4, 5, 6])
sub1lum = sub1.create_dataset("lum", data = testSnapDict[1]["lum"])
sub1totm = sub1.create_dataset("total mass", data = [6]) # data param must be in an array
sub1cent = sub1.create_dataset("central", data = [True])

for i in range(3):
    newgrp = f.create_group(str(i))
    newgrp.create_dataset("data", data=[i, i+1, i+2])

print(f["0"]["data"])