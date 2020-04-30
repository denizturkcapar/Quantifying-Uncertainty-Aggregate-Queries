from . import helpers


import datetime
import textdistance
import editdistance
import pandas as pd
import networkx as nx

from src.helpers import similarityMetrics


class bipartiteGraph(object):
    """

    Transforms the given file to a pandas dataframe object if it was not one already
    Assumption: Assumes that the data starts from the 1st row of given file, does not use seperators such as "," or ";"

    Input: 2 variables that will be measured for similarity
    Output: The desired similarity metric result
    """
    def similarity_edit(x,y):
        return editDistance(x,y,len(x),len(y))

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
            df = pd.read_csv(file, encoding='latin-1')
            return df
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

    Constructs a maximal bipartite graph of the given two tables

    Input: Any 2 files in any format
    Output: A Bipartite Graph with Maximal Weights
    """
    def updated_maximal_construct_graph(file_a, file_b):
        table_a_unprocessed = convert_df(file_a)
        table_b_unprocessed = convert_df(file_b)
        bipartite_graph = nx.Graph()
        
        table_a = make_dict(table_a_unprocessed)
        table_b = make_dict(table_b_unprocessed)
        
        i=0
        
        for key1, val1 in table_a.items():
            comp_point_1 = val1[0]

            id1 = str(key1) + '_' + str(comp_point_1) + '_1'
            for key2, val2 in table_b.items():
                comp_point_2 = val2[0]
                i+=1
                if i%100000 == 0:
                    print(str(round(100*i/len(table_a)/len(table_b),2))+'% complete')
                    
                #add value to identifier to distinguish two entries with different values
                
                id2 = str(key2) + '_' + str(comp_point_2) + '_2' 
                bipartite_graph.add_edge(id1, id2, weight=calc_max_weight_edit(str(comp_point_1), str(comp_point_2)))
                
                #edit distance and weight should be inv. prop.
                #also adding 1 to denom. to prevent divide by 0
                # add 1,2 to distinguish two key-value tuples belonging to different tables
                
        return bipartite_graph

    """

    Constructs a maximal bipartite graph of the given two tables

    Input: Any 2 files in any format
    Output: A Bipartite Graph with Minimal Weights
    """
    def updated_minimal_construct_graph(file_a, file_b):
        table_a_unprocessed = convert_df(file_a)
        table_b_unprocessed = convert_df(file_b)
        bipartite_graph = nx.Graph()
        
        table_a = make_dict(table_a_unprocessed)
        table_b = make_dict(table_b_unprocessed)
        
        i=0
        
        for key1, val1 in table_a.items():
            comp_point_1 = val1[0]

            id1 = str(key1) + '_' + str(comp_point_1) + '_1'
            for key2, val2 in table_b.items():
                comp_point_2 = val2[0]
                i+=1
                if i%100000 == 0:
                    print(str(round(100*i/len(table_a)/len(table_b),2))+'% complete')
                    
                #add value to identifier to distinguish two entries with different values
                
                id2 = str(key2) + '_' + str(comp_point_2) + '_2' 
                bipartite_graph.add_edge(id1, id2, weight=calc_min_weight_edit(str(comp_point_1), str(comp_point_2)))
                
                #edit distance and weight should be inv. prop.
                #also adding 1 to denom. to prevent divide by 0
                # add 1,2 to distinguish two key-value tuples belonging to different tables
                
        return bipartite_graph

    """

    Constructs a maximal bipartite graph of the given two tables according to the treshold similarity.
    The bipartite matching graph only includes those that have passed a certain similarity treshold.

    Input: Any 2 files in any format
    Output: A Bipartite Graph with Minimal Weights
    """
    # convert to lowercase
    def treshold_updated_maximal_construct_graph(file_a, file_b, treshold_decimal):
        table_a_unprocessed = convert_df(file_a)
        table_b_unprocessed = convert_df(file_b)
        bipartite_graph = nx.Graph()
        
        table_a = make_dict(table_a_unprocessed)
        table_b = make_dict(table_b_unprocessed)
        
        i=0
        
        for key1, val1 in table_a.items():
            comp_point_1 = val1[0]
          #  print(comp_point_1)
            id1 = str(key1) + '_'+ str(comp_point_1) + '_1'

            for key2, val2 in table_b.items():
                comp_point_2 = val2[0]
                dist = calc_max_weight_edit(str(comp_point_1).lower(),str(comp_point_2).lower())
                i+=1
                if i%100000 == 0:
                    print(str(round(100*i/len(table_a)/len(table_b),2))+'% complete')
                if dist >= treshold_decimal:
                  #  print(key1,key2,dist)
                    #add value to identifier to disitnguish two entries with different values
                    id2 = str(key2) + '_' + str(comp_point_2) + '_2' 
                    bipartite_graph.add_edge(id1, id2, weight=dist)
                    #edit distance and weight should be inv. prop.
                    #also adding 1 to denom. to prevent divide by 0
                    # add 1,2 to distinguish two key-value tuples belonging to different tables
                else:
                    continue
                
        return bipartite_graph
