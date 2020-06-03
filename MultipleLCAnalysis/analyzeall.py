import MultipleFiles as mf
import glob
import sys
import numpy as np

path = "/home/xjh0560/Supernova_Lightcurves/LC_Data/"
filelist = np.asarray(glob.glob("/home/xjh0560/Supernova_Lightcurves/LC_Data/sample_lc_v2/*.h5"))

args = sys.argv

print("Running")
mf.run_analysis_multi(filelist[int(args[1]):int(args[1])+2])