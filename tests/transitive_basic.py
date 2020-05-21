import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from src import transitive_closure
#from src import helpers

import datetime
import textdistance
import editdistance
import pandas as pd
import networkx as nx

table_a = transitive_closure.convert_df("table_a.csv")
table_b = transitive_closure.convert_df("table_b.csv")
repeat = transitive_closure.create_duplicates(table_b, "aaa", 3)
print("The duplicated table is as follows:\n", repeat, "\n")

now = datetime.datetime.now()
treshold_graph_maximal = transitive_closure.keycomp_treshold_updated_maximal_construct_graph(table_b, table_b, "aaa", 0.5)
timing_tresh = (datetime.datetime.now()-now).total_seconds()
print("---- Timing for Graph Construction with Treshold Constraint ----")
print(timing_tresh,"seconds")


"""
Outputs the matching that has the maximal weight for each edge in the bipartite graph
Input: A Bipartite Graph
Output: A set of matchings. Ex: {('journals/vldb/MedjahedBBNE03__2', '775457__1')}
"""
now = datetime.datetime.now()
matching_set = nx.algorithms.matching.max_weight_matching(treshold_graph_maximal)
timing_match = (datetime.datetime.now()-now).total_seconds()
print("---- Timing for Matching (Done on the graph constructed with the treshold constraint) ----")
print(timing_match,"seconds")
print("The number of edges in the graph is:", treshold_graph_maximal.number_of_edges(), "\n")


print("The Matching Set is:", matching_set, "\n")


out = transitive_closure.collapse(matching_set)
print("The cleaned matching is:", out, "\n")

last = transitive_closure.collapsed_dict(out)
print("The de-duplicated final result is:", last, "\n")
