from .context import src

import datetime
import textdistance
import editdistance
import pandas as pd
import networkx as nx

"""

Retrieves the keys that are going to be compared for the matching with the perfect mapping to evaluate the accuracy.

Input: Matching should be of type tuples inside a set.
Output: A tuple of matchings. Ex: ((idDBLP_1, idACM_1))
"""
def retrieve(matching):
    res2 = list(matching)
    res_tuple = []
    for i in res2:

        if int(i[0].split("_")[-1]) == 1:
            idACM = i[0].split("_")[0]
            idDBLP = i[1].split("_")[0]
            res_tuple.append((idDBLP, idACM))
            
        if int(i[0].split("_")[-1]) == 2:
            idACM = i[1].split("_")[0]
            idDBLP = i[0].split("_")[0]
            res_tuple.append((idDBLP, idACM))
            
    return res_tuple


import csv

def eval_matching(matching):
    f = open('DBLP-ACM_perfectMapping.csv', 'r', encoding = "ISO-8859-1")
    reader = csv.reader(f, delimiter=',', quotechar='"')
    matches = set()
    proposed_matches = set()

    tp = set()
    fp = set()
    fn = set()
    tn = set()

    for row in reader:
        matches.add((row[0],row[1]))

    for m in matching:
     #   print(m)
        proposed_matches.add(m)

        if m in matches:
            tp.add(m)
        else:
            fp.add(m)

    for m in matches:
        if m not in proposed_matches:
            fn.add(m)

    prec = len(tp)/(len(tp) + len(fp))
    rec = len(tp)/(len(tp) + len(fn))

    return {'false positive': 1-prec, 
            'false negative': 1-rec,
            'accuracy': 2*(prec*rec)/(prec+rec)}


#prints out the accuracy for 0.3 threshold
now = datetime.datetime.now()
out = eval_matching(retrieve(matching_set)) # retrieve() returns a list of tuples of DBLP2 ids and ACM ids.
timing = (datetime.datetime.now()-now).total_seconds()
print("----Accuracy----")
print(out)
print("---- Timing ----")
print(timing,"seconds")

#prints out the accuracy for 0.25 threshold
now = datetime.datetime.now()
out = eval_matching(retrieve(lowered_tresh_matching)) # retrieve() returns a list of tuples of DBLP2 ids and ACM ids.
timing_lower = (datetime.datetime.now()-now).total_seconds()
print("----Accuracy----")
print(out)
print("---- Timing ----")
print(timing_lower,"seconds")

#prints out accuracy for 0 treshold ()


