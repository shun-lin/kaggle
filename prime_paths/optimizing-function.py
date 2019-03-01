import pandas as pd
import numpy as np
import time
import pylab as pl
import math
import os
import pickle
import gzip
from operator import itemgetter
from matplotlib import collections  as mc
import copy
from random import randint
from itertools import *

def save_in_file_fast(arr, file_name):
    pickle.dump(arr, open(file_name, 'wb'))


def load_from_file_fast(file_name):
    return pickle.load(open(file_name, 'rb'))


def SieveOfEratosthenes(n):
    # Create a boolean array "prime[0..n]" and initialize
    #  all entries it as true. A value in prime[i] will
    # finally be false if i is Not a prime, else true.
    prime = [True for i in range(n + 1)]
    p = 2
    while (p * p <= n):

        # If prime[p] is not changed, then it is a prime
        if (prime[p] == True):

            # Update all multiples of p
            for i in range(p * 2, n + 1, p):
                prime[i] = False
        p += 1
    return prime

def get_primes():
    cache_path = OUTPUT_PATH + 'prime_list.pkl'
    if not os.path.isfile(cache_path):
        n = 200000
        prime = SieveOfEratosthenes(n)
        plist = []
        for p in range(2, n):
            if prime[p]:
                plist.append(p)
        save_in_file_fast(set(plist), cache_path)
    else:
        plist = load_from_file_fast(cache_path)

    return plist

# Globals
INPUT_PATH = 'input/'
OUTPUT_PATH = './'
CITIES = pd.read_csv('input/cities.csv')
PRIMES = get_primes()

all_ids = CITIES['CityId'].values
all_x = CITIES['X'].values
all_y = CITIES['Y'].values

CITIES_HASH = dict()
for i, id in enumerate(all_ids):
    CITIES_HASH[id] = (all_x[i], all_y[i])

def isPrime(num):
    return num in PRIMES

ORIGINAL = pd.read_csv('optimized_submission.csv')['Path'].values

def get_complete_score(tour):

    score = 0.0
    for i in range(0, len(tour)-1):
        p1 = CITIES_HASH[tour[i]]
        p2 = CITIES_HASH[tour[i+1]]
        stepSize = math.sqrt((p1[0] - p2[0]) * (p1[0] - p2[0]) + (p1[1] - p2[1]) * (p1[1] - p2[1]))
        if ((i + 1) % 10 == 0) and (tour[i] not in PRIMES):
            stepSize *= 1.1
        score += stepSize
    return score

def get_score(tour, start, end):
    score = 0.0
    for i in range(start, end):
        p1 = CITIES_HASH[tour[i]]
        p2 = CITIES_HASH[tour[i+1]]
        stepSize = math.sqrt((p1[0] - p2[0]) * (p1[0] - p2[0]) + (p1[1] - p2[1]) * (p1[1] - p2[1]))
        if ((i + 1) % 10 == 0) and (tour[i] not in PRIMES):
            stepSize *= 1.1
        score += stepSize
    return score

def swap(arr, i, j):
    temp = arr[i]
    arr[i] = arr[j]
    arr[j] = temp

def save_to_csv(arr):
    sub = pd.DataFrame(np.array(arr), columns = ["Path"])
    sub.to_csv('optimized_submission.csv', index=None)

def optimization2():
    numIter = 50000000
    iteration = 0
    modified = copy.deepcopy(ORIGINAL)
    total_reduction = 0.0
    last_saved_reduction = 0.0
    radius = 0
    while True:
        radius += 1
        print("radius increase to: " + str(radius))
        for step in range(9, len(ORIGINAL) - 1, 10):
            best_reduction = 0
            for i in [-radius, radius]:
                # save every 100000 iterations
                if iteration % 100000 == 0:
                    print("iteration: " + str(iteration))
                    if last_saved_reduction > total_reduction:
                        last_saved_reduction = total_reduction
                        print("saving to csv")
                        print("total reduction so far: " + str(total_reduction))
                        save_to_csv(modified)
                        print("score so far: " + str(get_complete_score(modified)))
                    else:
                        print("no improvement")
                iteration += 1
                index = step + i
                if (index < len(ORIGINAL) and index > 0):
                    start = min(step, index) - 1
                    end = min(max(step, index) + 1, len(ORIGINAL) - 1)
                    original_score = get_score(modified, start, end)
                    swap(modified, step, index)
                    modified_score = get_score(modified, start, end)
                    score_diff = modified_score - original_score
                    if score_diff < best_reduction:
                        best_reduction = score_diff
                        best_swap = [step, index]
                    # swap back
                    swap(modified,  step, index)
            if (best_reduction < 0):
                swap(modified, best_swap[0], best_swap[1])
                total_reduction += best_reduction
    return modified, total_reduction

def optimization3():
    numIter = 50000000
    iteration = 0
    modified = copy.deepcopy(ORIGINAL)
    total_reduction = 0.0
    last_saved_reduction = 0.0
    radius = 9
    last_checked_iteration = 0
    while True:
        radius += 1
        print("radius increase to: " + str(radius))
        for origin in range(9, len(ORIGINAL) - 1, 10):
            best_reduction = 0
            start = max(origin - radius, 0)
            end = min(origin + radius, len(ORIGINAL) - 1)
            path_to_check = copy.deepcopy(modified[start:end])
            original_score = get_score(modified, max(start - 1, 0), min(end + 1, len(ORIGINAL) - 1))
            best_combo = copy.deepcopy(path_to_check)
            for combo in permutations(path_to_check, len(path_to_check)):
                iteration += 1
                for i in range(len(combo)):
                    modified[origin - radius + i] = combo[i]
                modified_score = get_score(modified, max(start - 1, 0), min(end + 1, len(ORIGINAL) - 1))
                score_diff = modified_score - original_score
                if score_diff < best_reduction:
                    best_reduction = score_diff
                    best_combo = copy.deepcopy(combo)
            total_reduction += best_reduction
            for j in range(len(best_combo)):
                modified[origin - radius + j] = best_combo[j]
            # save appx. every 100000 iterations
            if iteration > last_checked_iteration + 100000:
                last_checked_iteration = iteration
                print("iteration: " + str(iteration))
                if last_saved_reduction > total_reduction:
                    last_saved_reduction = total_reduction
                    print("saving to csv")
                    print("total reduction so far: " + str(total_reduction))
                    save_to_csv(modified)
                    print("score so far: " + str(get_complete_score(modified)))
                else:
                    print("no improvement")
    return modified, total_reduction

print("starting ...")
naive_modified3, total_reduction = optimization3()
print("end result")
print("improve score by: " + str(-total_reduction))
print("end!")