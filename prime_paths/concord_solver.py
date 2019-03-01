from concorde.tsp import TSPSolver
from matplotlib import collections  as mc
import numpy as np
import pandas as pd
import time
import pylab as pl
import os
import pickle
import gzip
from operator import itemgetter
import math

cities = pd.read_csv('input/cities.csv')

# Instantiate solver
solver = TSPSolver.from_data(
    [int(x * 1000) for x in cities.X],
    [int(y * 1000) for y in cities.Y],
    norm="EUC_2D"
)

print("starting ...")
t = time.time()
tour_data = solver.solve(verbose = False) # solve() doesn't seem to respect time_bound for certain values?
print(time.time() - t)
print(tour_data.found_tour)

pd.DataFrame({'Path': np.append(tour_data.tour,[0])}).to_csv('submission_concord_full.csv', index=False)
print("end ...")\