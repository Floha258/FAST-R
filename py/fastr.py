'''
This is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This software is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this source.  If not, see <http://www.gnu.org/licenses/>.
'''

from collections import defaultdict
from collections import OrderedDict
from sklearn_extra.cluster import KMedoids
from pyclustering.cluster.clarans import clarans
import math
import os
import pickle
import random
import sys
import time

from functools import reduce
import numpy as np

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.random_projection import johnson_lindenstrauss_min_dim
from sklearn.random_projection import SparseRandomProjection

import lsh

"""
This file implements FAST-R test suite reduction algorithms.
"""


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# utility function to load test suite
def loadTestSuite(input_file, bbox=False, k=5):
    TS = defaultdict()
    with open(input_file) as fin:
        tcID = 1
        for tc in fin:
            if bbox:
                TS[tcID] = tc[:-1]
            else:
                TS[tcID] = set(tc[:-1].split())
            tcID += 1
    shuffled = list(TS.keys())
    random.shuffle(shuffled)
    newTS = OrderedDict()
    for key in shuffled:
        newTS[key] = TS[key]
    if bbox:
        newTS = lsh.kShingles(TS, k)
    return newTS


# store signatures on disk for future re-use
def storeSignatures(input_file, sigfile, hashes, bbox=False, k=5):
    with open(sigfile, "w") as sigfile:
        with open(input_file) as fin:
            tcID = 1
            for tc in fin:
                if bbox:
                    # shingling
                    tc_ = tc[:-1]
                    tc_shingles = set()
                    for i in range(len(tc_) - k + 1):
                        tc_shingles.add(hash(tc_[i:i + k]))

                    sig = lsh.tcMinhashing((tcID, set(tc_shingles)), hashes)
                else:
                    tc_ = tc[:-1].split()
                    sig = lsh.tcMinhashing((tcID, set(tc_)), hashes)
                for hash_ in sig:
                    sigfile.write(hash_)
                    sigfile.write(" ")
                sigfile.write("\n")
                tcID += 1


# load stored signatures
def loadSignatures(input_file):
    sig = {}
    start = time.time()
    with open(input_file, "r") as fin:
        tcID = 1
        for tc in fin:
            sig[tcID] = [i.strip() for i in tc[:-1].split()]
            tcID += 1
    return sig, time.time() - start


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


# FAST-PW (pairwise comparison with candidate set)
def fast_pw(input_file, r, b, bbox=False, k=5, memory=False, B=0):
    n = r * b  # number of hash functions

    hashes = [lsh.hashFamily(i) for i in range(n)]

    if memory:
        test_suite = loadTestSuite(input_file, bbox=bbox, k=k)
        # generate minhashes signatures
        mh_t = time.time()
        tcs_minhashes = {tc[0]: lsh.tcMinhashing(tc, hashes)
                         for tc in test_suite.items()}
        mh_time = time.time() - mh_t
        ptime_start = time.time()

    else:
        # loading input file and generating minhashes signatures
        sigfile = input_file.replace(".txt", ".sig")
        sigtimefile = "{}_sigtime.txt".format(input_file.split(".")[0])
        if not os.path.exists(sigfile):
            mh_t = time.time()
            storeSignatures(input_file, sigfile, hashes, bbox, k)
            mh_time = time.time() - mh_t
            with open(sigtimefile, "w") as fout:
                fout.write(repr(mh_time))
        else:
            with open(sigtimefile, "r") as fin:
                mh_time = eval(fin.read().replace("\n", ""))

        ptime_start = time.time()
        tcs_minhashes, load_time = loadSignatures(sigfile)

    tcs = set(tcs_minhashes.keys())

    # budget B modification
    if B == 0:
        B = len(tcs)

    BASE = 0.5
    SIZE = int(len(tcs) * BASE) + 1

    bucket = lsh.LSHBucket(tcs_minhashes.items(), b, r, n)

    prioritized_tcs = [0]

    # First TC

    selected_tcs_minhash = lsh.tcMinhashing((0, set()), hashes)
    first_tc = random.choice(list(tcs_minhashes.keys()))
    for i in range(n):
        if tcs_minhashes[first_tc][i] < selected_tcs_minhash[i]:
            selected_tcs_minhash[i] = tcs_minhashes[first_tc][i]
    prioritized_tcs.append(first_tc)
    tcs -= set([first_tc])
    del tcs_minhashes[first_tc]

    iteration, total = 0, float(len(tcs_minhashes))
    while len(tcs_minhashes) > 0:
        iteration += 1
        if iteration % 100 == 0:
            sys.stdout.write("  Progress: {}%\r".format(
                round(100 * iteration / total, 2)))
            sys.stdout.flush()

        if len(tcs_minhashes) < SIZE:
            bucket = lsh.LSHBucket(tcs_minhashes.items(), b, r, n)
            SIZE = int(SIZE * BASE) + 1

        sim_cand = lsh.LSHCandidates(bucket, (0, selected_tcs_minhash),
                                     b, r, n)
        filtered_sim_cand = sim_cand.difference(prioritized_tcs)
        candidates = tcs - filtered_sim_cand

        if len(candidates) == 0:
            selected_tcs_minhash = lsh.tcMinhashing((0, set()), hashes)
            sim_cand = lsh.LSHCandidates(bucket, (0, selected_tcs_minhash),
                                         b, r, n)
            filtered_sim_cand = sim_cand.difference(prioritized_tcs)
            candidates = tcs - filtered_sim_cand
            if len(candidates) == 0:
                candidates = tcs_minhashes.keys()

        selected_tc, max_dist = random.choice(tuple(candidates)), -1
        for candidate in tcs_minhashes:
            if candidate in candidates:
                dist = lsh.jDistanceEstimate(
                    selected_tcs_minhash, tcs_minhashes[candidate])
                if dist > max_dist:
                    selected_tc, max_dist = candidate, dist

        for i in range(n):
            if tcs_minhashes[selected_tc][i] < selected_tcs_minhash[i]:
                selected_tcs_minhash[i] = tcs_minhashes[selected_tc][i]

        prioritized_tcs.append(selected_tc)

        # select budget B
        if len(prioritized_tcs) >= B + 1:
            break

        tcs -= set([selected_tc])
        del tcs_minhashes[selected_tc]

    ptime = time.time() - ptime_start

    max_ts_size = sum((1 for line in open(input_file)))
    return mh_time, ptime, prioritized_tcs[1:max_ts_size]


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# FAST-f (for any input function f, i.e., size of candidate set)
def fast_(input_file, selsize, r, b, bbox=False, k=5, memory=False, B=0):
    n = r * b  # number of hash functions

    hashes = [lsh.hashFamily(i) for i in range(n)]

    if memory:
        test_suite = loadTestSuite(input_file, bbox=bbox, k=k)
        # generate minhashes signatures
        mh_t = time.time()
        tcs_minhashes = {tc[0]: lsh.tcMinhashing(tc, hashes)
                         for tc in test_suite.items()}
        mh_time = time.time() - mh_t
        ptime_start = time.time()

    else:
        # loading input file and generating minhashes signatures
        sigfile = input_file.replace(".txt", ".sig")
        sigtimefile = "{}_sigtime.txt".format(input_file.split(".")[0])
        if not os.path.exists(sigfile):
            mh_t = time.time()
            storeSignatures(input_file, sigfile, hashes, bbox, k)
            mh_time = time.time() - mh_t
            with open(sigtimefile, "w") as fout:
                fout.write(repr(mh_time))
        else:
            with open(sigtimefile, "r") as fin:
                mh_time = eval(fin.read().replace("\n", ""))

        ptime_start = time.time()
        tcs_minhashes, load_time = loadSignatures(sigfile)

    tcs = set(tcs_minhashes.keys())

    # budget B modification
    if B == 0:
        B = len(tcs)

    BASE = 0.5
    SIZE = int(len(tcs) * BASE) + 1

    bucket = lsh.LSHBucket(tcs_minhashes.items(), b, r, n)

    prioritized_tcs = [0]

    # First TC

    selected_tcs_minhash = lsh.tcMinhashing((0, set()), hashes)
    first_tc = random.choice(list(tcs_minhashes.keys()))
    for i in range(n):
        if tcs_minhashes[first_tc][i] < selected_tcs_minhash[i]:
            selected_tcs_minhash[i] = tcs_minhashes[first_tc][i]
    prioritized_tcs.append(first_tc)
    tcs -= set([first_tc])
    del tcs_minhashes[first_tc]

    iteration, total = 0, float(len(tcs_minhashes))
    while len(tcs_minhashes) > 0:
        iteration += 1
        if iteration % 100 == 0:
            sys.stdout.write("  Progress: {}%\r".format(
                round(100 * iteration / total, 2)))
            sys.stdout.flush()

        if len(tcs_minhashes) < SIZE:
            bucket = lsh.LSHBucket(tcs_minhashes.items(), b, r, n)
            SIZE = int(SIZE * BASE) + 1

        sim_cand = lsh.LSHCandidates(bucket, (0, selected_tcs_minhash),
                                     b, r, n)
        filtered_sim_cand = sim_cand.difference(prioritized_tcs)
        candidates = tcs - filtered_sim_cand

        if len(candidates) == 0:
            selected_tcs_minhash = lsh.tcMinhashing((0, set()), hashes)
            sim_cand = lsh.LSHCandidates(bucket, (0, selected_tcs_minhash),
                                         b, r, n)
            filtered_sim_cand = sim_cand.difference(prioritized_tcs)
            candidates = tcs - filtered_sim_cand
            if len(candidates) == 0:
                candidates = tcs_minhashes.keys()

        to_sel = min(selsize(len(candidates)), len(candidates))
        selected_tc_set = random.sample(tuple(candidates), to_sel)

        for selected_tc in selected_tc_set:
            for i in range(n):
                if tcs_minhashes[selected_tc][i] < selected_tcs_minhash[i]:
                    selected_tcs_minhash[i] = tcs_minhashes[selected_tc][i]

            prioritized_tcs.append(selected_tc)

            # select budget B
            if len(prioritized_tcs) >= B + 1:
                break

            tcs -= set([selected_tc])
            del tcs_minhashes[selected_tc]

        # select budget B
        if len(prioritized_tcs) >= B + 1:
            break

    ptime = time.time() - ptime_start

    max_ts_size = sum((1 for line in open(input_file)))
    return mh_time, ptime, prioritized_tcs[1:max_ts_size]


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Preparation + utils

# compute euclidean distance
def euclideanDist(v, w):
    d = 0

    for k in v.keys():
        if k not in w.keys():
            d += v[k] ** 2
        else:
            d += (v[k] - w[k]) ** 2

    for k in w.keys():
        if k not in v.keys():
            d += w[k] ** 2

    return math.sqrt(d)


# Preparation phase for FAST++ and FAST-CS
def preparation(inputFile, dim=0):
    vectorizer = HashingVectorizer()  # compute "TF"
    testCases = [line.rstrip("\n") for line in open(inputFile)]
    testSuite = vectorizer.fit_transform(testCases)

    # dimensionality reduction
    if dim <= 0:
        e = 0.5  # epsilon in jl lemma
        dim = johnson_lindenstrauss_min_dim(len(testCases), eps=e)
    srp = SparseRandomProjection(n_components=dim)
    projectedTestSuite = srp.fit_transform(testSuite)

    # map sparse matrix to dict
    TS = []
    for i in range(len(testCases)):
        tc = {}
        for j in projectedTestSuite[i].nonzero()[1]:
            tc[j] = projectedTestSuite[i, j]
        TS.append(tc)

    return TS


# Alternate Preparation phase for own algos
def preparationAlt(inputFile, dim=0):
    vectorizer = HashingVectorizer()  # compute "TF"
    testCases = [line.rstrip("\n") for line in open(inputFile)]
    testSuite = vectorizer.fit_transform(testCases)

    # dimensionality reduction
    if dim <= 0:
        e = 0.5  # epsilon in jl lemma
        dim = johnson_lindenstrauss_min_dim(len(testCases), eps=e)
    srp = SparseRandomProjection(n_components=dim)
    projectedTestSuite = srp.fit_transform(testSuite)

    # map sparse matrix to dict
    TS = []
    for i in range(len(testCases)):
        tc = {}
        for j in projectedTestSuite[i].nonzero()[1]:
            tc[j] = projectedTestSuite[i, j]
        TS.append(tc)

    return projectedTestSuite


def sparseToDict(medoids, B):
    TS = []
    for i in range(B):
        tc = {}
        for j in medoids[i].nonzero()[1]:
            tc[j] = medoids[i, j]
        TS.append(tc)

    return TS


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# FAST++ Reduction phase
def reductionPlusPlus(TS, B):
    reducedTS = []

    # distance to closest center
    D = defaultdict(lambda: float('Inf'))
    # select first center randomly
    selectedTC = random.randint(0, len(TS) - 1)
    reducedTS.append(selectedTC + 1)
    D[selectedTC] = 0

    while len(reducedTS) < B:
        # k-means++ tc reductionCS
        norm = 0
        for tc in range(len(TS)):
            if D[tc] != 0:
                dist = euclideanDist(TS[tc], TS[selectedTC])
                dist *= dist
                if dist < D[tc]:
                    D[tc] = dist
            norm += D[tc]

        # safe exit point (if all distances are 0)
        # (but not all test cases have been selected)
        if norm == 0:
            extraTCS = list(set(range(1, len(TS) + 1)) - set(reducedTS))
            random.shuffle(extraTCS)
            reducedTS.extend(extraTCS[:B - len(reducedTS)])
            break

        c = 0
        coinToss = random.random() * norm
        for tc, dist in D.items():
            if coinToss < c + dist:
                reducedTS.append(tc + 1)
                D[tc] = 0
                break
            c += dist

    return reducedTS


"""
def initMedoids(TS, B):
    np.random.seed(1)
    samples = np.random.choice(TS, size=B, replace=False)
    print('Samples = {}'.format(samples))
    return samples


def computeDistance(TS, medoids, p):
    m = len(TS)
    medoids_shape = medoids.shape
    print('m = ' + str(m) + 'shape = ' + str(medoids_shape))
    if (len(medoids_shape)) == 1:
        medoids = medoids.reshape((1, len(medoids)))
    k = len(medoids)
    S = np.empty((m, k))
    for i in range(m):
        print(str(i))
        print(str(TS[i:]))
        d_i = np.linalg.norm(TS[i:] - medoids, ord=p, axis=1)
        S[i, :] = d_i ** p
    return S

def assignLabels(S):
    return np.argmin(S, axis=1)

def updateMedoids(TS, medoids, p):
    S = computeDistance(TS, medoids, p)
    labels = assignLabels(S)

    out_medoids = medoids

    for i in set(labels):
        avg_dissamilarity = np.sum(computeDistance(TS, medoids[i], p))
        cluster_points = TS[medoids == i]

        for datap in cluster_points:
            new_medoid = datap
            new_dissamilarity = np.sum(computeDistance(TS, datap, 2))

            if new_dissamilarity < avg_dissamilarity:
                avg_dissamilarity = new_dissamilarity
                out_medoids[i] = datap

    return out_medoids

def k_medoids(TS, B, starting_medoids=None, max_steps=np.inf):
    if starting_medoids is None:
        medoids = initMedoids(TS, B)
    else:
        medoids = starting_medoids
    converged = False
    labels = np.zeros(len(TS))
    i = 1
    while (not converged) and (i <= max_steps):
        old_medoids = medoids.copy
        S = computeDistance(TS, medoids, 2)
        labels = assignLabels(S)
        medoids = updateMedoids(TS, medoids, 2)
        converged = hasConverged(old_medoids, medoids)
        i += 1
    return medoids

"""


def kMedoids(D, k, tmax=100):
    # determine dimensions of distance matrix D
    """
    tsArray = np.array()

    for tc in TS:
        tcArray = np.array()
        for key in tc.keys():
            vector = np.array(key, tc[key])
            tcArray.__add__(vector)
        tsArray.__add__(tcArray)
        """

    m, n = D.shape

    if k > n:
        raise Exception('too many medoids')

    # find a set of valid initial cluster medoid indices since we
    # can't seed different clusters with two points at the same location
    valid_medoid_inds = set(range(n))
    invalid_medoid_inds = set([])
    rs, cs = np.where(D == 0)
    # the rows, cols must be shuffled because we will keep the first duplicate below
    index_shuf = list(range(len(rs)))
    np.random.shuffle(index_shuf)
    rs = rs[index_shuf]
    cs = cs[index_shuf]
    for r, c in zip(rs, cs):
        # if there are two points with a distance of 0...
        # keep the first one for cluster init
        if r < c and r not in invalid_medoid_inds:
            invalid_medoid_inds.add(c)
    valid_medoid_inds = list(valid_medoid_inds - invalid_medoid_inds)

    if k > len(valid_medoid_inds):
        raise Exception('too many medoids (after removing {} duplicate points)'.format(
            len(invalid_medoid_inds)))

    # randomly initialize an array of k medoid indices
    M = np.array(valid_medoid_inds)
    np.random.shuffle(M)
    M = np.sort(M[:k])

    # create a copy of the array of medoid indices
    Mnew = np.copy(M)

    # initialize a dictionary to represent clusters
    C = {}
    for t in range(tmax):
        # determine clusters, i. e. arrays of data indices
        J = np.argmin(D[:, M], axis=1)
        for kappa in range(k):
            C[kappa] = np.where(J == kappa)[0]
        # update cluster medoids
        for kappa in range(k):
            J = np.mean(D[np.ix_(C[kappa], C[kappa])], axis=1)
            # Fix for the low-idx bias by J.Nagele (2019):
            shuffled_idx = np.arange(len(J))
            np.random.shuffle(shuffled_idx)
            j = shuffled_idx[np.argmin(J[shuffled_idx])]
            Mnew[kappa] = C[kappa][j]
        np.sort(Mnew)
        # check for convergence
        if np.array_equal(M, Mnew):
            break
        M = np.copy(Mnew)
    else:
        # final update of cluster memberships
        J = np.argmin(D[:, M], axis=1)
        for kappa in range(k):
            C[kappa] = np.where(J == kappa)[0]

    # return results
    return M, C


# FAST-medoids reduction
def reductionMedoids(TS, B):
    D = np.array(TS)
    print(str)
    kmedoids = KMedoids(n_clusters=B).fit(TS)
    # return sparseToDict(kmedoids.cluster_centers_, B)

    reducedTS = []
    for medoid in kmedoids.cluster_centers_:
        i = 0
        for tc in TS:
            if not medoid.tolil != tc.tolil:
                reducedTS.append(i)
        i += 1

    return reducedTS


# reduction for clustering with CLARANS
def reductionCLARANS(TS, B):
    # data = []
    # for tc in TS:
    #    tcarr = []
    #    for entry in tc:
    #        tcarr.append(entry)
    #    data.append(tcarr)

    data = TS.tolil().data.tolist()
    # print("TS: {}".format(str(TS)))
    # print("data: {}".format(str(data)))
    print(str(type(data)))
    print(str(type(data[0])))
    clusters = clarans(data, B, 500, 400)
    clusters.process()

    reducedTS = []
    print("Medoids: {}, OG TS: {}".format(str(clusters.get_medoids()), str(TS)))
    for medoid in clusters.get_medoids():
        i = 0
        for tc in TS:
            if not medoid != tc.tolil:
                reducedTS.append(i)
            i += 1

    return reducedTS


# FAST++ test suite reduction algorithm
# Returns: preparation time, reduction time, reduced test suite
def fastPlusPlus(inputFile, dim=0, B=0, memory=True):
    if memory:
        t0 = time.time()
        TS = preparation(inputFile, dim=dim)
        t1 = time.time()
        pTime = t1 - t0
    else:
        rpFile = inputFile.replace(".txt", ".rp")
        if not os.path.exists(rpFile):
            t0 = time.time()
            TS = preparation(inputFile, dim=dim)
            t1 = time.time()
            pTime = t1 - t0
            pickle.dump((pTime, TS), open(rpFile, "wb"))
        else:
            pTime, TS = pickle.load(open(rpFile, "rb"))

    if B <= 0:
        B = len(TS)

    t2 = time.time()
    reducedTS = reductionPlusPlus(TS, B)
    t3 = time.time()
    sTime = t3 - t2

    return pTime, sTime, reducedTS


# FAST-Medoids test suite reduction algorithm
# Returns: preparation time, reduction time, reduced test suite
def fastMedoids(inputFile, dim=0, B=0, memory=True):
    if memory:
        t0 = time.time()
        TS = preparationAlt(inputFile, dim=dim)
        t1 = time.time()
        pTime = t1 - t0
    else:
        rpFile = inputFile.replace(".txt", ".rp")
        if not os.path.exists(rpFile):
            t0 = time.time()
            TS = preparation(inputFile, dim=dim)
            t1 = time.time()
            pTime = t1 - t0
            pickle.dump((pTime, TS), open(rpFile, "wb"))
        else:
            pTime, TS = pickle.load(open(rpFile, "rb"))

    if B <= 0:
        B = len(TS)

    t2 = time.time()
    reducedTS = reductionMedoids(TS, B)
    t3 = time.time()
    sTime = t3 - t2

    return pTime, sTime, reducedTS


# FAST-CLARANS test suite reduction algorithm
# Returns: preparation time, reduction time, reduced test suite
def fastCLARANS(inputFile, dim=0, B=0, memory=True):
    if memory:
        t0 = time.time()
        TS = preparationAlt(inputFile, dim=dim)
        t1 = time.time()
        pTime = t1 - t0
    else:
        rpFile = inputFile.replace(".txt", ".rp")
        if not os.path.exists(rpFile):
            t0 = time.time()
            TS = preparation(inputFile, dim=dim)
            t1 = time.time()
            pTime = t1 - t0
            pickle.dump((pTime, TS), open(rpFile, "wb"))
        else:
            pTime, TS = pickle.load(open(rpFile, "rb"))

    if B <= 0:
        B = len(TS)

    t2 = time.time()
    reducedTS = reductionCLARANS(TS, B)
    t3 = time.time()
    sTime = t3 - t2

    return pTime, sTime, reducedTS


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# FAST-CS

# FAST-CS Reduction phase
def reductionCS(TS, B):
    reducedTS = []

    # compute center of mass
    centerOfMass = defaultdict(float)
    for tc in TS:
        for k, v in tc.items():
            centerOfMass[k] += v
    # normalize
    for k in centerOfMass.keys():
        centerOfMass[k] /= len(TS)

    # compute distances
    D = defaultdict(float)
    norm = 0
    for tc in range(len(TS)):
        dist = euclideanDist(TS[tc], centerOfMass)
        D[tc] = dist * dist
        norm += D[tc]

    # compute probabilities of being sampled
    P = []
    if norm != 0:
        p = 1.0 / (2 * len(TS))
        for tc in range(len(TS)):
            P.append(p + D[tc] / (2 * norm))
    else:
        P = [1.0 / len(TS)] * len(TS)

    # numeric error: when sum of P != 1
    P[random.randint(0, len(TS) - 1)] += 1.0 - sum(P)

    # proportional sampling
    reducedTS = list(np.random.choice(list(range(1, len(TS) + 1)), size=B, p=P, replace=False))

    return reducedTS


# FAST-CS test suite reduction algorithm
# Returns: preparation time, reduction time, reduced test suite
def fastCS(inputFile, dim=0, B=0, memory=True):
    if memory:
        t0 = time.time()
        TS = preparation(inputFile, dim=dim)
        t1 = time.time()
        pTime = t1 - t0
    else:
        rpFile = inputFile.replace(".txt", ".rp")
        if not os.path.exists(rpFile):
            t0 = time.time()
            TS = preparation(inputFile, dim=dim)
            t1 = time.time()
            pTime = t1 - t0
            pickle.dump((pTime, TS), open(rpFile, "wb"))
        else:
            pTime, TS = pickle.load(open(rpFile, "rb"))

    if B <= 0:
        B = len(TS)

    t2 = time.time()
    reducedTS = reductionCS(TS, B)
    t3 = time.time()
    sTime = t3 - t2

    return pTime, sTime, reducedTS
