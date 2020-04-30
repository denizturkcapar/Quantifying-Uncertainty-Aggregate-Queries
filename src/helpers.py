import datetime
import textdistance
import editdistance
import pandas as pd
import networkx as nx


def editDistance(str1, str2, m, n): 
  
    # If first string is empty, the only option is to 
    # insert all characters of second string into first 
    if m == 0: 
         return n 
  
    # If second string is empty, the only option is to 
    # remove all characters of first string 
    if n == 0: 
        return m 
  
    # If last characters of two strings are same, nothing 
    # much to do. Ignore last characters and get count for 
    # remaining strings. 
    if str1[m-1]== str2[n-1]: 
        return editDistance(str1, str2, m-1, n-1) 
  
    # If last characters are not same, consider all three 
    # operations on last character of first string, recursively 
    # compute minimum cost for all three operations and take 
    # minimum of three values. 
    return 1 + min(editDistance(str1, str2, m, n-1),    # Insert 
                   editDistance(str1, str2, m-1, n),    # Remove 
                   editDistance(str1, str2, m-1, n-1)    # Replace 
                   ) 


def jaccard_similarity(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection) / union

