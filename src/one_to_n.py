import re
import sys

import datetime
import textdistance
import editdistance
import pandas as pd
import networkx as nx
import collections


"""

Transforms the given file to a pandas dataframe object if it was not one already
Assumption: Assumes that the data starts from the 1st row of given file, does not use seperators such as "," or ";"

Input: Any file
Output: A pandas dataframe object
"""
def convert_df(file):
    if isinstance(file, pd.DataFrame):
        return file
    else:
        df = pd.read_csv(file)
        return df
# changed latin encoding, it was giving an error with table_a example

"""

Duplicates the dataframe that is assigned the "1" labeled data frame in the "1-n" matching scheme.
The entries that are duplicated is seperated from each other using the "_1", "_2", ..., "_num" scheme.

Input: 
  -- df: A pandas dataframe, 
  -- column: The column in the dataframe that will be marked with _1, _2, ..., _num
  -- num: A user inputted amount of duplicates to be found in the "n" labeled table.
Output: A pandas dataframe object
"""
def create_duplicates(df, col, num):

	# Repeat without index
	df_repeated = pd.concat([df]*num,ignore_index=True)

	n = len(df)

	for i, row in df_repeated.iterrows():
		df_repeated.loc[i, col] = row[col]+'_'+str(i//n)

#	print(df_repeated)
	return df_repeated



"""

Calculates maximum weight for the matching

Input: keys from 2 tables
Output: weight for each matching to be used in the weight part of constructing the graph
"""
def calc_max_weight(key1, key2):
    weight = textdistance.jaccard(key1,key2) #this library's implementation is slower than jaccard_similarity()
    return weight

"""

Calculates minimum weight for the matching

Input: keys from 2 tables
Output: weight for each matching to be used in the weight part of constructing the graph
"""
def calc_min_weight(key1, key2):
    weight = (-1)/(1+textdistance.jaccard(key1,key2)) #this library's implementation is slower than jaccard_similarity()
    return weight

"""

Calculates maximum weight for the matching

Input: keys from 2 tables
Output: weight for each matching to be used in the weight part of constructing the graph
"""
def calc_max_weight_edit(key1, key2):
    weight = (1)/(1+editdistance.eval(key1,key2))
    return weight

"""

Calculates minimum weight for the matching

Input: keys from 2 tables
Output: weight for each matching to be used in the weight part of constructing the graph
"""
def calc_min_weight_edit(key1, key2):
    weight = (-1)/(1+editdistance.eval(key1,key2))
    return weight



"""

Converts the dataframe into dictionary for better accuracy matching of pairs. 
Assumption: The data has headers in the first row (description of what that column describes)

Input: Any file
Output: A dictionary in the form col1:col2 matching
"""
def make_dict(file):
    V = list(file.to_dict('list').values())
    keys = V[0]
    values = zip(*V[1:])
    table = dict(zip(keys,values))
    return table


"""

Constructs a maximal bipartite graph of the given two tables according to the treshold similarity.
The bipartite matching graph only includes those that have passed a certain similarity treshold.

Input: Any 2 files in any format
Output: A Bipartite Graph with Minimal Weights
"""
# convert to lowercase
def treshold_updated_maximal_construct_graph(file_one, file_n, treshold_decimal):
    table_a_unprocessed = convert_df(file_one)
    table_b_unprocessed = convert_df(file_n)
    bipartite_graph = nx.Graph()
    
    table_a_unprocessed = create_duplicates(table_a_unprocessed, "aa", 3) # Assuming that the user inputs 3 duplicates

    table_a = make_dict(table_a_unprocessed)
    table_b = make_dict(table_b_unprocessed)
    
    i=0
    
    for key1, val1 in table_a.items():
        comp_point_1 = key1

        id1 = str(key1) + '_'+ str(val1) + '_1'
        for key2, val2 in table_b.items():
            comp_point_2 = key2
            dist = calc_max_weight_edit(str(comp_point_1).lower(),str(comp_point_2).lower())
            i+=1
            if i%100000 == 0:
                print(str(round(100*i/len(file_one)/len(file_n),2))+'% complete')
            if dist <= treshold_decimal:
                #add value to identifier to disitnguish two entries with different values
                id2 = str(key2) + '_' + str(val1) + '_2' + "_" + str(dist)
                bipartite_graph.add_edge(id1, id2, weight=dist)
                #edit distance and weight should be inv. prop.
                #also adding 1 to denom. to prevent divide by 0
                # add 1,2 to distinguish two key-value tuples belonging to different tables
            else:
                continue
            
    return bipartite_graph

"""
Retrieves the keys that are going to be compared for the matching with the perfect mapping to evaluate the accuracy.

Input: Matching should be of type tuples inside a set.
Output: A tuple of matchings. Ex: ((idDBLP_1, idACM_1))
"""

def collapse(matching_set):
	res2 = list(matching_set)
	res_tuple = []
	for i in res2:
	#	print(i)
		if i[0].split("_")[1].isdigit() == True:

			if int(i[0].split("_")[3]) == 1:
				idACM = i[0].split("_")[0] + "_1"
				idDBLP = i[1].split("_")[0]
				res_tuple.append((idDBLP, idACM))
			if int(i[0].split("_")[3]) == 2:
				idACM = i[1].split("_")[0] + "_1"
				idDBLP = i[0].split("_")[0]
				res_tuple.append((idDBLP, idACM))

		if i[1].split("_")[1].isdigit() == True:

			if int(i[0].split("_")[2]) == 1:
				idACM = i[1].split("_")[0] + "_1"
				idDBLP = i[0].split("_")[0]
				res_tuple.append((idDBLP, idACM))
			if int(i[0].split("_")[2]) == 2:
				idACM = i[1].split("_")[0] + "_1"
				idDBLP = i[0].split("_")[0]
				res_tuple.append((idDBLP, idACM))

	return res_tuple

def collapsed_dict(res):
	out = collections.defaultdict(list)
	for (val, key) in res:
		print(val,key)
	#	if key not in out:
	#		out[key] = val
	#	else:
		out[key].append(str(val))
	return out
"""
		if int(i[0].split("_")[2]) == 1:
			idACM = i[0].split("_")[0]
			idDBLP = i[1].split("_")[0]
			res_tuple.append((idDBLP, idACM))
        if int(i[0].split("_")[2]) == 2:
        	idACM = i[1].split("_")[0]
        	idDBLP = i[0].split("_")[0]
        	res_tuple.append((idDBLP, idACM))

    return res_tuple

			
			if int(i[0].split("_")[3]) == 2:
				idACM = i[1].split("_")[0]
				idDBLP = i[0].split("_")[0]
				res_tuple.append((idDBLP, idACM))
					

			
"""


