import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from src import core
from src import helpers

import datetime
import textdistance
import editdistance
import pandas as pd
import networkx as nx

bipartite_graph_maximal = core.updated_maximal_construct_graph("table_a.csv","table_b.csv")
#print(bipartite_graph_maximal.edges.data())
bipartite_graph_minimal = core.updated_minimal_construct_graph("table_a.csv", "table_b.csv")
bipartite_graph_minimal.edges.data()

#nx.algorithms.matching.max_weight_matching(bipartite_graph_maximal)
print(nx.algorithms.bipartite.matching.maximum_matching(bipartite_graph_minimal))

# Load the data for processing
# Sticking to the convention of table_a and table_b naming that we previously used for generalization purposes
table_a = core.convert_df("ACM.csv")

table_b = core.convert_df("DBLP2.csv")

now = datetime.datetime.now()
graph_maximal = core.updated_maximal_construct_graph(table_a, table_b)
timing_bg = (datetime.datetime.now()-now).total_seconds()
print("---- Timing for Graph Construction ----")
print(timing_bg,"seconds")
graph_maximal.number_of_edges()


now = datetime.datetime.now()
treshold_graph_maximal = core.treshold_updated_maximal_construct_graph(table_a, table_b, 0.3)
timing_tresh = (datetime.datetime.now()-now).total_seconds()
print("---- Timing for Graph Construction with Treshold Constraint ----")
print(timing_tresh,"seconds")
#treshold_graph_maximal.number_of_edges()

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
treshold_graph_maximal.number_of_edges()



perfect_mapping = core.convert_df("DBLP-ACM_perfectMapping.csv")
row = perfect_mapping.shape[0]
print("Number of entries in the perfect mapping file:", row)
print("Difference between the entries in perfect mapping and the matching set:", len(perfect_mapping) - len(matching_set))

lowered_tresh_graph = core.treshold_updated_maximal_construct_graph(table_a, table_b, 0.25)
lowered_tresh_matching = nx.algorithms.matching.max_weight_matching(lowered_tresh_graph)


print("The number of edges that the graph with a treshold of 0.25 contains is:", lowered_tresh_graph.number_of_edges())


