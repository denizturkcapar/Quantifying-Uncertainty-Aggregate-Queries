import re
import sys

import datetime
import textdistance
import editdistance
import pandas as pd
import networkx as nx
import collections
import math



def string_sets(word):
    return(set(word.lower().split()))

def calc_jaccard(amazon_titles, google_titles):
    set_google = string_sets(google_titles)
    set_amazon = string_sets(amazon_titles)
    return len(set_amazon.intersection(set_google))/len(set_amazon.union(set_google))


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

def lat_convert_df(file):
    if isinstance(file, pd.DataFrame):
        return file
    else:
        df = pd.read_csv(file, encoding='latin-1')
        return df

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
		df_repeated.loc[i, col] = str(row[col])+'_'+str(i//n)

#	print(df_repeated)
	return df_repeated

"""

Calculates maximum weight for the matching

Input: keys from 2 tables
Output: weight for each matching to be used in the weight part of constructing the graph
"""
def calc_max_weight(key1, key2):

    weight = (1)/(1+calc_jaccard(key1,key2)) #this library's implementation is slower than jaccard_similarity()
    # print(weight)
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
    # print(key1)
    # print(key2)
    # print(editdistance.eval(key1,key2))
    weight = (1)/(1+editdistance.eval(key1,key2))
    # print(weight)
    return weight

"""

Calculates minimum weight for the matching

Input: keys from 2 tables
Output: weight for each matching to be used in the weight part of constructing the graph
"""
def calc_min_weight_edit(key1, key2):
    weight = (1)/(1+editdistance.eval(key1,key2))
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
*****************************************************************
MAXIMAL MATCHING
*****************************************************************
Constructs a maximal bipartite graph of the given two tables according to the treshold similarity.
The bipartite matching graph only includes those that have passed a certain similarity treshold.
The similarity metric takes into account the **keys** in this implementation

Input: Any 2 files in any format
Output: A Bipartite Graph with Maximal Weights
"""
def keycomp_treshold_updated_maximal_construct_graph(file_one, file_n, col_to_dup, treshold_decimal):
    table_a_unprocessed = convert_df(file_one)
    table_b_unprocessed = convert_df(file_n)
    bipartite_graph = nx.Graph()
    
    table_a_unprocessed = create_duplicates(table_a_unprocessed, col_to_dup, 3) # Assuming that the user inputs 3 duplicates

    table_a = make_dict(table_a_unprocessed)
    table_b = make_dict(table_b_unprocessed)
    
    i=0
    
    for key1, val1 in table_a.items():
        comp_point_1 = key1.split("_")[0]

        id1 = str(key1) + '_'+ str(val1) + '_1'
        for key2, val2 in table_b.items():

            comp_point_2 = key2.split("_")[0]
            dist = calc_max_weight_edit(str(comp_point_1).lower(),str(comp_point_2).lower())
            i+=1
            if i%100000 == 0:
                print(str(round(100*i/len(file_one)/len(file_n),2))+'% complete')
            if dist <= treshold_decimal:
                # print(comp_point_1, comp_point_2, dist)
                #add value to identifier to disitnguish two entries with different values
                id2 = str(key2) + '_' + str(val2) + '_2'
                bipartite_graph.add_edge(id1, id2, weight=dist)
                #edit distance and weight should be inv. prop.
                #also adding 1 to denom. to prevent divide by 0
                # add 1,2 to distinguish two key-value tuples belonging to different tables
            else:
                continue
            
    return bipartite_graph

"""

Constructs a maximal bipartite graph of the given two tables according to the treshold similarity.
The bipartite matching graph only includes those that have passed a certain similarity treshold.
The similarity metric takes into account the **values** in this implementation

Input: Any 2 files in any format
Output: A Bipartite Graph with Maximal Weights
"""
def valcomp_treshold_updated_maximal_construct_graph(file_one, file_n, col_to_dup, treshold_decimal):
    table_a_unprocessed = convert_df(file_one)
    table_b_unprocessed = convert_df(file_n)
    bipartite_graph = nx.Graph()
    
    table_a_unprocessed = create_duplicates(table_a_unprocessed, col_to_dup, 3) # Assuming that the user inputs 3 duplicates

    table_a = make_dict(table_a_unprocessed)
    table_b = make_dict(table_b_unprocessed)

    i=0
    
    for key1, val1 in table_a.items():
        comp_point_1 = val1[0].split("_")[0]

        id1 = str(key1) + '_'+ str(comp_point_1) + "_" + str(val1[3]) + '_1'
        for key2, val2 in table_b.items():

            comp_point_2 = val2[0]
            dist = calc_jaccard(str(comp_point_1).lower(),str(comp_point_2).lower())
            i+=1
            # print("first is: ", comp_point_1, "second is:", comp_point_2, "distance is:", dist)
            if i%100000 == 0:
                print(str(round(100*i/len(file_one)/len(file_n),2))+'% complete')
            if dist >= treshold_decimal:
                
                #add value to identifier to disitnguish two entries with different values
                id2 = str(key2) + '_' + str(comp_point_2) + "_" + str(val2[3]) + '_2'
                bipartite_graph.add_edge(id1, id2, weight=dist)
                #edit distance and weight should be inv. prop.
                #also adding 1 to denom. to prevent divide by 0
                # add 1,2 to distinguish two key-value tuples belonging to different tables
            else:
                continue
            
    return bipartite_graph

"""
*******************************************************************
MINIMAL MATCHING
*******************************************************************
Constructs a minimal bipartite graph of the given two tables according to the treshold similarity.
The bipartite matching graph only includes those that have passed a certain similarity treshold.
The similarity metric takes into account the **keys** in this implementation

Input: Any 2 files in any format
Output: A Bipartite Graph with Minimal Weights
"""
def keycomp_treshold_updated_minimal_construct_graph(file_one, file_n, col_to_dup, treshold_decimal):
    table_a_unprocessed = convert_df(file_one)
    table_b_unprocessed = convert_df(file_n)
    bipartite_graph = nx.Graph()
    
    table_a_unprocessed = create_duplicates(table_a_unprocessed, col_to_dup, 3) # Assuming that the user inputs 3 duplicates

    table_a = make_dict(table_a_unprocessed)
    table_b = make_dict(table_b_unprocessed)
    
    i=0
    
    for key1, val1 in table_a.items():
        comp_point_1 = key1

        id1 = str(key1) + '_'+ str(val1) + '_1'
        for key2, val2 in table_b.items():
            print(key2, val2)
            comp_point_2 = key2
            dist = calc_min_weight_edit(str(comp_point_1).lower(),str(comp_point_2).lower())
            i+=1
            if i%100000 == 0:
                print(str(round(100*i/len(table_a)/len(table_b),2))+'% complete')
            if dist >= treshold_decimal:
            #add value to identifier to disitnguish two entries with different values
                id2 = str(key2) + '_' + str(val2) + '_2'
                print("id is:", id2)
                bipartite_graph.add_edge(id1, id2, weight=dist)
            #edit distance and weight should be inv. prop.
            #also adding 1 to denom. to prevent divide by 0
            # add 1,2 to distinguish two key-value tuples belonging to different tables
            else:
                continue
    return bipartite_graph

def simpler_min_graph_construct(file_one, file_n, col_to_dup):
    table_a_unprocessed = convert_df(file_one)
    table_b_unprocessed = convert_df(file_n)
    bipartite_graph = nx.Graph()
    
    table_a_unprocessed = create_duplicates(table_a_unprocessed, col_to_dup, 3) # Assuming that the user inputs 3 duplicates

    table_a = make_dict(table_a_unprocessed)
    table_b = make_dict(table_b_unprocessed)
    
    i=0
    
    for key1, val1 in table_a.items():
        comp_point_1 = key1

        id1 = str(key1) + '_'+ str(val1) + '_1'
        for key2, val2 in table_b.items():
            print(key2, val2)
            comp_point_2 = key2
            dist = calc_max_weight_edit(str(comp_point_1).lower(),str(comp_point_2).lower())
            i+=1
            if i%100000 == 0:
                print(str(round(100*i/len(table_a)/len(table_b),2))+'% complete')
            # if dist <= treshold_decimal:
            #add value to identifier to disitnguish two entries with different values
            id2 = str(key2) + '_' + str(val2) + '_2'
            print("id is:", id2)
            bipartite_graph.add_edge(id1, id2, weight=dist)
            #edit distance and weight should be inv. prop.
            #also adding 1 to denom. to prevent divide by 0
            # add 1,2 to distinguish two key-value tuples belonging to different tables
            # else:
            #     continue
    return bipartite_graph

"""
(Appropriated inputs from keycomp_treshold_updated_minimal_construct_graph only for idBLP ACM dataset.)
Constructs a minimal bipartite graph of the given two tables according to the treshold similarity.
The bipartite matching graph only includes those that have passed a certain similarity treshold.
The similarity metric takes into account the **values** in this implementation

Input: Any 2 files in any format
Output: A Bipartite Graph with Minimal Weights
"""
def valcomp_treshold_updated_minimal_construct_graph(file_one, file_n, treshold_decimal):
    table_a_unprocessed = convert_df(file_one)
    table_b_unprocessed = convert_df(file_n)
    bipartite_graph = nx.Graph()

    table_a_unprocessed = create_duplicates(table_a_unprocessed, "id", 3) # Assuming that the user inputs 3 duplicates
    
    table_a = make_dict(table_a_unprocessed)
    table_b = make_dict(table_b_unprocessed)
    
    i=0
    
    for key1, val1 in table_a.items():
        comp_point_1 = val1[0]
      #  print(comp_point_1)
        id1 = str(key1) + '_'+ str(comp_point_1) + '_1'

        for key2, val2 in table_b.items():
            comp_point_2 = val2[0]
            dist = calc_max_weight(str(comp_point_1).lower(),str(comp_point_2).lower())
            i+=1
            if i%100000 == 0:
                print(str(round(100*i/len(table_a)/len(table_b),2))+'% complete')
            if dist >= treshold_decimal:
                print(id1,id2,dist)
                #add value to identifier to disitnguish two entries with different values
                id2 = str(key2) + '_' + str(comp_point_2) + '_2' 
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

		if i[0].split("_")[1].isdigit() == True:

			if i[0].split("_")[3] == "1":
				idACM = i[0].split("_")[0] + "_1"
				idDBLP = i[1].split("_")[0]
				res_tuple.append((idDBLP, idACM))
			if i[0].split("_")[3] == "2":
				idACM = i[1].split("_")[0] + "_1"
				idDBLP = i[0].split("_")[0]
				res_tuple.append((idDBLP, idACM))

		if i[1].split("_")[1].isdigit() == True:

			if i[0].split("_")[2] == "1":
				idACM = i[1].split("_")[0] + "_1"
				idDBLP = i[0].split("_")[0]
				res_tuple.append((idDBLP, idACM))
			if i[0].split("_")[2] == "2":
				idACM = i[1].split("_")[0] + "_1"
				idDBLP = i[0].split("_")[0]
				res_tuple.append((idDBLP, idACM))

	return res_tuple


def collapse2(matching_set):
    res2 = list(matching_set.items())
    res_tuple = []
    for i in res2:

        if i[0].split("_")[1].isdigit() == True:

            if i[0].split("_")[3] == "1":
                idACM = i[0].split("_")[0] + "_1"
                idDBLP = i[1].split("_")[0]
                res_tuple.append((idDBLP, idACM))
            if i[0].split("_")[3] == "2":
                idACM = i[1].split("_")[0] + "_1"
                idDBLP = i[0].split("_")[0]
                res_tuple.append((idDBLP, idACM))

        if i[1].split("_")[1].isdigit() == True:

            if i[0].split("_")[2] == "1":
                idACM = i[1].split("_")[0] + "_1"
                idDBLP = i[0].split("_")[0]
                res_tuple.append((idDBLP, idACM))
            if i[0].split("_")[2] == "2":
                idACM = i[1].split("_")[0] + "_1"
                idDBLP = i[0].split("_")[0]
                res_tuple.append((idDBLP, idACM))

    return res_tuple
"""

Collapse the results further to a dictionary format. I've made it optional as a separate function for now.
Although it is quite possible to merge this function with collapsed() function

Input: The output received from collapsed() function
Output: A dictionary that has the key as the de-duplicated entries and the values as the "n" table values that they got matched
"""
def collapsed_dict(res):

    out = collections.defaultdict(set)
    for (val, key) in res:
        # print("\n\n", val, key)
        out[key].add(str(val))
    return out

# def collapsed_minimal_dict(res):
#     out = {}
#     for (val, key) in res:
#         out[key] = [val]
"""

Identify the matchings that have the 1:n matching nature. Since this example uses a dataset that has a 1:1 nature
we don't expect to see a lot of 1:n findings from our algorithm

Input: key:[values] mappings of the matchings
Output: key:[values] mappings of the matchings that have a list length that is greater than 1.
"""
def more_than_one(dic):
	out2 = []
	for key, val in dic.items():
		if len(val) > 1:
			out2.append((key,val))
	return(out2)

"""
*************************************************************************
Fall Quarter Updates
*************************************************************************
Maximal and Minimal Matching for SUM Operation
"""
"""
Helper function for create_val_lookup. 
Returns a dictionary of specified column index
"""
def make_dict_specific_col(file, col_index):
    V = list(file.to_dict('list').values())
    keys = V[0]
    values = V[col_index]
    table = dict(zip(keys,values))
    return table


def create_val_lookup(file_1, file_n, col_index):
    lookup = make_dict_specific_col(file_1, col_index)
    lookup_update = make_dict_specific_col(file_n, col_index)
    lookup.update(lookup_update)
    return lookup

def SUM_result_with_uncertainties(max_out, min_out, lookup):
    max_cumulative_sum = {}    
    min_cumulative_sum = {}

    # Calculations for maximum interval using maximal matching results
    for key, vals in max_out.items():


        true_key = key.split("_")[0]
        # print(true_key)
        # print(int(lookup[true_key]))
        # print(true_key, val)
        if true_key not in max_cumulative_sum:
            # print(lookup[true_key])
            # print(max_cumulative_sum[true_key])
            # print(lookup[true_key])
            max_cumulative_sum[true_key] = int(lookup[true_key])
            continue
        else:
            max_cumulative_sum[true_key] += int(lookup[true_key])
            # if true_key not in formal_output:
            #     formal_output[true_key] = []
            continue
        for val in vals:
            if val in lookup:
                max_cumulative_sum[true_key] += int(lookup[val])
        # else:
        #     # if true_key not in formal_output:
        #     #     formal_output[true_key] = []
        #     continue

    #Calculations for minimum interval using minimal matching results
    for key, val in min_out.items():

        true_key = key.split("_")[0]
        if true_key not in min_cumulative_sum:
            min_cumulative_sum[true_key] = int(lookup[true_key])
            # if true_key not in formal_output:
            #     formal_output[true_key] = []
            continue
        else:
            min_cumulative_sum[true_key] += int(lookup[true_key])
            # if true_key not in formal_output:
            #     formal_output[true_key] = []
            continue
        # else:
        #     continue
        for val in vals:
            if val in lookup:
                max_cumulative_sum[true_key] += int(lookup[val])
    return(min_cumulative_sum, max_cumulative_sum)

def MIN_result_with_uncertainties(max_out, min_out, lookup):
    max_cumulative_sum = {}    
    min_cumulative_sum = {}

    # Calculations for maximum interval using maximal matching results
    for key, vals in max_out.items():


        true_key = key.split("_")[0]
        # print(true_key)
        # print(int(lookup[true_key]))
        # print(true_key, val)
        if true_key not in max_cumulative_sum:
            # print(lookup[true_key])
            # print(max_cumulative_sum[true_key])
            # print(lookup[true_key])
            max_cumulative_sum[true_key] = int(lookup[true_key])
            continue
        else:
            max_cumulative_sum[true_key] += int(lookup[true_key])
            # if true_key not in formal_output:
            #     formal_output[true_key] = []
            continue
        for val in vals:
            if val in lookup:
                max_cumulative_sum[true_key] += int(lookup[val])
        # else:
        #     # if true_key not in formal_output:
        #     #     formal_output[true_key] = []
        #     continue

    #Calculations for minimum interval using minimal matching results
    for key, val in min_out.items():

        true_key = key.split("_")[0]
        if true_key not in min_cumulative_sum:
            min_cumulative_sum[true_key] = int(lookup[true_key])
            # if true_key not in formal_output:
            #     formal_output[true_key] = []
            continue
        else:
            min_cumulative_sum[true_key] += int(lookup[true_key])
            # if true_key not in formal_output:
            #     formal_output[true_key] = []
            continue
        # else:
        #     continue
        for val in vals:
            if val in lookup:
                max_cumulative_sum[true_key] += int(lookup[val])
    return(min_cumulative_sum, max_cumulative_sum)


def MAX_result_with_uncertainties(max_out, min_out, lookup):
    max_cumulative_sum = {}    
    min_cumulative_sum = {}

    # Calculations for maximum interval using maximal matching results
    for key, vals in max_out.items():


        true_key = key.split("_")[0]
        # print(true_key)
        # print(int(lookup[true_key]))
        # print(true_key, val)
        if true_key not in max_cumulative_sum:
            # print(lookup[true_key])
            # print(max_cumulative_sum[true_key])
            # print(lookup[true_key])
            max_cumulative_sum[true_key] = int(lookup[true_key])
            continue
        else:
            max_cumulative_sum[true_key] += int(lookup[true_key])
            # if true_key not in formal_output:
            #     formal_output[true_key] = []
            continue
        for val in vals:
            if val in lookup:
                max_cumulative_sum[true_key] += int(lookup[val])
        # else:
        #     # if true_key not in formal_output:
        #     #     formal_output[true_key] = []
        #     continue

    #Calculations for minimum interval using minimal matching results
    for key, val in min_out.items():

        true_key = key.split("_")[0]
        if true_key not in min_cumulative_sum:
            min_cumulative_sum[true_key] = int(lookup[true_key])
            # if true_key not in formal_output:
            #     formal_output[true_key] = []
            continue
        else:
            min_cumulative_sum[true_key] += int(lookup[true_key])
            # if true_key not in formal_output:
            #     formal_output[true_key] = []
            continue
        # else:
        #     continue
        for val in vals:
            if val in lookup:
                max_cumulative_sum[true_key] += int(lookup[val])
    return(min_cumulative_sum, max_cumulative_sum)


def COUNT_result_with_uncertainties(max_out, min_out, lookup):
    max_cumulative_sum = {}    
    min_cumulative_sum = {}

    # Calculations for maximum interval using maximal matching results
    for key, vals in max_out.items():


        true_key = key.split("_")[0]
        # print(true_key)
        # print(int(lookup[true_key]))
        # print(true_key, val)
        if true_key not in max_cumulative_sum:
            # print(lookup[true_key])
            # print(max_cumulative_sum[true_key])
            # print(lookup[true_key])
            max_cumulative_sum[true_key] = int(lookup[true_key])
            continue
        else:
            max_cumulative_sum[true_key] += int(lookup[true_key])
            # if true_key not in formal_output:
            #     formal_output[true_key] = []
            continue
        for val in vals:
            if val in lookup:
                max_cumulative_sum[true_key] += int(lookup[val])
        # else:
        #     # if true_key not in formal_output:
        #     #     formal_output[true_key] = []
        #     continue

    #Calculations for minimum interval using minimal matching results
    for key, val in min_out.items():

        true_key = key.split("_")[0]
        if true_key not in min_cumulative_sum:
            min_cumulative_sum[true_key] = int(lookup[true_key])
            # if true_key not in formal_output:
            #     formal_output[true_key] = []
            continue
        else:
            min_cumulative_sum[true_key] += int(lookup[true_key])
            # if true_key not in formal_output:
            #     formal_output[true_key] = []
            continue
        # else:
        #     continue
        for val in vals:
            if val in lookup:
                max_cumulative_sum[true_key] += int(lookup[val])
    return(min_cumulative_sum, max_cumulative_sum)



def form_formal_output(min_final, max_final, min_cumulative_sum, max_cumulative_sum):
    formal_output = {}
    # print(min_cumulative_sum)
    # print(max_cumulative_sum)

    # Format output as following: {USA : [ [US], [USAA, US, UK], [5], [100] ], ...}

    for key, vals in min_final.items():
        true_key = key.split("_")[0]
        # if true_key in formal_output:
        #     formal_output[true_key].append([min_out[true_key]])
        # else:
        formal_output[true_key] = [vals]
    # print(formal_output)
    # print(formal_output['AU'].append(['19']))
    # print(formal_output['AU'].append(['20']))
    # print(formal_output)
    for key, vals in max_final.items():
        true_key = key.split("_")[0]
        if true_key in formal_output:
            formal_output[true_key].append([vals])
            continue
        else:
            formal_output[true_key] = []
    
    for key, val in min_cumulative_sum.items():
        true_key = key.split("_")[0]
        if true_key in formal_output:
            formal_output[true_key].append([min_cumulative_sum[true_key]])

    for key, val in max_cumulative_sum.items():
        true_key = key.split("_")[0]
        if true_key in formal_output:
            formal_output[true_key].append([max_cumulative_sum[true_key]])

    return formal_output


def SUM_edit_edge_weight(bip_graph):
    for u,v,d in bip_graph.edges(data=True):
        val_tuple_1 = u.split("_")
        for i in val_tuple_1:
            if i.startswith("("):
                val1 = i.split(",")[0][1:]
        val_tuple_2 = v.split("_")
        for j in val_tuple_2:
            if j.startswith("("):
                val2 = j.split(",")[0][1:]
        d['weight'] = int(val1) + int(val2)
        # print("val1 is: ", val1)
        # print("val2 is: ", val2)
        # print("data is: ", d)

    return bip_graph


def COUNT_edit_edge_weight(bip_graph):
    for u,v,d in bip_graph.edges(data=True):
        d['weight'] = 1
    return bip_graph
