import time
import numpy
import pandas
import plotnine
import scipy
import sklearn

#%matplotlib widget 

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

data = pandas.read_csv('/Users/Elen/Documents/PhD/MSc_Thesis/BNC2014Baby/MFTE_Evaluation_BNC2014_Results_merged.csv', keep_default_na=False)

