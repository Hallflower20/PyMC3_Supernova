import glob
import numpy as np
path = "/home/xjh0560/Supernova_Lightcurves/LC_Data/"
filelist = np.asarray(glob.glob("/home/xjh0560/Supernova_Lightcurves/LC_Data/sample_lc_v2/*.h5"))

for i in range(len(filelist)):
    if(i%2 == 0):
        fbase = open("base.sh", "r")
        base = fbase.read()
        fnew = open("runFile" + str(i) + ".sh", "w")
        fnew.write(base)
        fnew.write("\n")
        fnew.write('#SBATCH --output="jobout'+str(i)+'"\n#SBATCH --error="joberr'+str(i)+'"\n')
        fbase2 = open("base2.sh", "r")
        base2 = fbase2.read()
        fnew.write(base2)
        fnew.write("\npython /home/xjh0560/GitHub/PyMC3_Supernova/MultipleLCAnalysis/analyzeall.py " + str(i) + "\n")
        fnew.write("echo Done")
        fnew.close()