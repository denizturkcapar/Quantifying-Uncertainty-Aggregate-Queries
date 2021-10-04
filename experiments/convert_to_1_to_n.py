import sys
from os import path
import pandas as pd
import re
import csv
import datetime
import matplotlib.transforms as transforms
# import matplotlib.axes.Axes as ax
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import experiment_funcs
import one_to_n
from Matching import core
from Matching import analyze
from Matching import matcher
from Matching import core_scholar
import textdistance
import editdistance
from collections import defaultdict

# Google - 1's table
# Amazon - N's table
def modified_1_to_n_perfect_mapping_google_amazon(perf_match_file, new_filename):
	# Convert files to pandas df
	perf_match_table = one_to_n.lat_convert_df(perf_match_file)

	# record_counts = perf_match_table.pivot_table(index=['idAmazon'], aggfunc='size')
	# Make a dictionary of duplicated records info
	# This is needed for deleting these records info from perfect match dataset
	perf_mapping_dict_1_to_n = {}
	mapping_records1 = perf_match_table.to_records(index=False)
	mapping_result1 = list(mapping_records1)
	duplicates_info = defaultdict(list)
	for amazon, google in mapping_result1:
		if google in perf_mapping_dict_1_to_n:
			duplicates_info[google].append(amazon)
		else:
			perf_mapping_dict_1_to_n[google] = amazon

	# print(duplicates_info)
	# Delete duplicate k matchings from perfect match dataset
	for key, vals in duplicates_info.items():
		for value in vals:
			indexNames = perf_match_table[(perf_match_table['idGoogleBase'] == key) & (perf_match_table['idAmazon'] == value)].index
			# Delete these row indexes from dataFrame
			perf_match_table.drop(indexNames , inplace=True)

	perf_match_table.to_csv(new_filename, index=False)

# Abt = 1's table
# Buy = N's table (Max n=2)
def modified_1_to_n_perfect_mapping_abt_buy(perf_match_file, new_filename):
	# Convert files to pandas df
	perf_match_table = one_to_n.lat_convert_df(perf_match_file)
	# Make a dictionary of duplicated records info
	# This is needed for deleting these records info from perfect match dataset
	perf_mapping_dict_1_to_n = {}
	mapping_records1 = perf_match_table.to_records(index=False)
	mapping_result1 = list(mapping_records1)
	duplicates_info = defaultdict(list)
	for abt, buy in mapping_result1:
		if abt in perf_mapping_dict_1_to_n:
			duplicates_info[abt].append(buy)
		else:
			perf_mapping_dict_1_to_n[abt] = buy
	# Delete duplicate k matchings from perfect match dataset
	for key, vals in duplicates_info.items():
		for value in vals:
			indexNames = perf_match_table[(perf_match_table['idAbt'] == key) & (perf_match_table['idBuy'] == value)].index
			# Delete these row indexes from dataFrame
			perf_match_table.drop(indexNames , inplace=True)
	perf_match_table.to_csv(new_filename, index=False)

# Scholar = 1's table
# DBLP - n's table (max n = 20)
def modified_1_to_n_perfect_mapping_dblp_scholar(perf_match_file, new_filename):
	# Convert files to pandas df
	perf_match_table = one_to_n.lat_convert_df(perf_match_file)

	# Make a dictionary of duplicated records info
	# This is needed for deleting these records info from perfect match dataset
	perf_mapping_dict_1_to_n = {}
	mapping_records1 = perf_match_table.to_records(index=False)
	mapping_result1 = list(mapping_records1)
	duplicates_info = defaultdict(list)
	for scholar, dblp in mapping_result1:
		if scholar in perf_mapping_dict_1_to_n:
			duplicates_info[scholar].append(dblp)
		else:
			perf_mapping_dict_1_to_n[scholar] = dblp
	# Delete duplicate k matchings from perfect match dataset
	for key, vals in duplicates_info.items():
		for value in vals:
			indexNames = perf_match_table[(perf_match_table['idScholar'] == key) & (perf_match_table['idDBLP'] == value)].index
			# Delete these row indexes from dataFrame
			perf_match_table.drop(indexNames , inplace=True)

	perf_match_table.to_csv(new_filename, index=False)


modified_1_to_n_perfect_mapping_google_amazon('../Amazon-GoogleProducts/Amzon_GoogleProducts_perfectMapping.csv', '../Amazon-GoogleProducts/modified_perfect_mapping_google_amazon.csv')

modified_1_to_n_perfect_mapping_dblp_scholar('../DBLP-Scholar/DBLP-Scholar_perfectMapping.csv', '../DBLP-Scholar/modified_perfect_mapping_dblp_schoalr.csv')

modified_1_to_n_perfect_mapping_abt_buy('../Abt-Buy/abt_buy_perfectMapping.csv', '../Abt-Buy/modified_perfect_mapping_abt_buy.csv')






