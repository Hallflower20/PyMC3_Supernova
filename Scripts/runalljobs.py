import os
import glob
import numpy as np
path = "/home/xjh0560/Supernova_Lightcurves/LC_Data/"
filelist = np.asarray(glob.glob("/home/xjh0560/Supernova_Lightcurves/LC_Data/sample_lc_v2/*.h5"))

for i in range(len(filelist)):
    if(i%2 == 0):
        os.system("sbatch runFile" + str(i) + ".sh")

print("Done")