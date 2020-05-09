import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from src import one_to_n
#from src import helpers

import datetime
import textdistance
import editdistance
import pandas as pd
import networkx as nx

table_a = one_to_n.convert_df("table_a.csv")
table_b = one_to_n.convert_df("table_b.csv")
repeat = one_to_n.create_duplicates(table_a, "aa", 3)
print(repeat)

now = datetime.datetime.now()
treshold_graph_maximal = one_to_n.treshold_updated_maximal_construct_graph(table_a, table_b, 0.5)
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
print("The number of edges in the graph is:", treshold_graph_maximal.number_of_edges())
#print(matching_set)


out = one_to_n.collapse(matching_set)
print(out)
last = one_to_n.collapsed_dict(out)
print(last)
