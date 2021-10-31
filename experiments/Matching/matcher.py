from Matching import core
import editdistance
import math
import random
from collections import defaultdict
import re

"""
Calculates maximum weight for the matching

Input: keys from 2 tables
Output: weight for each matching to be used in the weight part of constructing the graph
"""
def calc_max_weight_edit(key1, key2, distance_func):
    # print("JACCARD DIST:", distance_func(key1,key2))
    weight = (1)/(1+distance_func(key1,key2))
    # print(weight)
    return weight

def calc_max_weight_jaccard(key1, key2, distance_func):
    # print("JACCARD DIST:", distance_func(key1,key2))
    weight = (1)/(1+distance_func(key1,key2))
    # print(weight)
    return weight

def matcher(d1, d2, distance_fn, sampler_fn, sample_size=100):

	match = []	

	for e1 in d1:
		min_dist = math.inf
		min_dist_id = None
		for e2 in sampler_fn(d2, sample_size):
			dist = distance_fn(e1['name'],e2['name'])
			if dist < min_dist:
				min_dist = dist
				min_dist_id = e2['name']
				sum_total = int(e1['age']) + int(e2['age'])
		match.append((e1['name'],min_dist_id, sum_total))

	return match

def matcher_dup(d1, d2, distance_fn, sampler_fn, sample_size=100):

	match = []
	for e1 in d1:
		# Note that d1 is always the duplicated table
		# So the entries of names need to cleaned for the "_number" adjustment during the matching
		cleaned_e1 = e1['name'].split("_")[0]
		min_dist = math.inf
		min_dist_id = None
		for e2 in sampler_fn(d2, sample_size):
			dist = distance_fn(cleaned_e1,e2['name'])
			if dist < min_dist:
				min_dist = distance
				min_dist_id = e2['name']
		sum_total = int(e1['age']) + int(e2['age'])
		match.append((e1['name'],min_dist_id, sum_total))

	return match

# Use only 1 matcher function, give num_match as a variable to get matches
def matcher_updated(num_matches, is_max, d1, d2, distance_fn, sampler_fn, required_distance, sample_size=100):

	# match = []
	match_map = defaultdict(list)
	for e1 in d1:

		for e2 in sampler_fn(d2, sample_size):
			distance = calc_max_weight_edit(str(e1['name']).lower(), str(e2['name']).lower(), distance_fn)
#			print("M1: ", e1['name'], "M2: ", e2['name'],"DIST: ", distance, "REQ:", required_distance)
			if distance >= required_distance:
				sum_total = int(e1['age']) + int(e2['age'])
				match_map[e1['name']].append((e1['name'],e2['name'], sum_total))
				# match.append((e1['name'],e2['name'], sum_total))
	for k,v in match_map.items():
		match_map[k] = sorted(v,key=lambda x: x[-1], reverse=True)

	if is_max == True:
		res = extract_top_n(match_map,num_matches)
	else:
		res = extract_bottom_1(match_map)
	return res

# Use only 1 matcher function, give num_match as a variable to get matches
def realdata_matcher_updated(num_matches, is_max, d1, d2, distance_fn, sampler_fn, required_distance, sample_size=100):
	res_without_sum = []
	# match = []
	match_map = defaultdict(list)
	# print("DF1: ", d1)
	# print("DF2 ", d2)
	trim = re.compile(r'[^\d.,]+')
	for e1 in d1:

		for e2 in sampler_fn(d2, sample_size):
			# print("M1: ", e1['name'], "M2: ", e2['name'])
			distance = calc_max_weight_jaccard(str(e1['name']).lower(), str(e2['name']).lower(), distance_fn)
			# print("M1: ", e1['name'], "M2: ", e2['name'],"DIST: ", distance, "REQ:", required_distance)
			# if distance != 0:
			# 	print("M1: ", e1['name'], "M2: ", e2['name'], "DIST: ", distance)
			if distance <= required_distance:
				val1 = trim.sub('', e1['price'])
				val2 = trim.sub('', e2['price'])
				# val1 = re.sub("[^0-9]", "", e1['price'])
				# val2 =re.sub("[^0-9]", "", e2['price'])
				# print("M1: ", e1['name'], "M2: ", e2['name'], e1['price'], e2['price'], "val1", val1, "val2", val2,"DIST: ", distance)
				sum_total = float(val1) + float(val2)
				match_map[e1['name']].append((e1['name'],e2['name'], sum_total))
				# match.append((e1['name'],e2['name'], sum_total))

				res_without_sum.append((e1['id'], e2['id']))

	for k,v in match_map.items():
		match_map[k] = sorted(v,key=lambda x: x[-1], reverse=True)

	if is_max == True:
		res = extract_top_n(match_map,num_matches)
	else:
		res = extract_bottom_1(match_map)
	return res, res_without_sum


def realdata_matcher_count(num_matches, is_max, d1, d2, distance_fn, sampler_fn, required_distance, sample_size=100):
	res_without_count = []
	# match = []
	match_map = defaultdict(list)
	# print("DF1: ", d1)
	# print("DF2 ", d2)
	trim = re.compile(r'[^\d.,]+')
	for e1 in d1:

		for e2 in sampler_fn(d2, sample_size):
			# print("M1: ", e1['name'], "M2: ", e2['name'])
			distance = calc_max_weight_jaccard(str(e1['name']).lower(), str(e2['name']).lower(), distance_fn)
			# print("M1: ", e1['name'], "M2: ", e2['name'],"DIST: ", distance, "REQ:", required_distance)
			# if distance != 0:
			# 	print("M1: ", e1['name'], "M2: ", e2['name'], "DIST: ", distance)
			if distance >= required_distance:
				count_total = 1
				match_map[e1['name']].append((e1['name'],e2['name'], count_total))
				# match.append((e1['name'],e2['name'], sum_total))

				res_without_count.append((e1['id'], e2['id']))

	for k,v in match_map.items():
		match_map[k] = sorted(v,key=lambda x: x[-1], reverse=True)

	if is_max == True:
		res = extract_top_n(match_map,num_matches)
	else:
		res = extract_bottom_1(match_map)
	return res, res_without_count

def realdata_matcher_count_variation(num_matches, is_max, d1, d2, distance_fn, sampler_fn, required_distance, filter_condition, sample_size=100):
	res_without_count = []
	# match = []
	match_map = defaultdict(list)
	# print("DF1: ", d1)
	# print("DF2 ", d2)
	trim = re.compile(r'[^\d.,]+')
	for e1 in d1:

		for e2 in sampler_fn(d2, sample_size):
			# print("M1: ", e1['name'], "M2: ", e2['name'])
			distance = calc_max_weight_jaccard(str(e1['name']).lower(), str(e2['name']).lower(), distance_fn)
			# print("M1: ", e1['name'], "M2: ", e2['name'],"DIST: ", distance, "REQ:", required_distance)
			# if distance != 0:
			# 	print("M1: ", e1['name'], "M2: ", e2['name'], "DIST: ", distance)
			# val1 = trim.sub('', e1['price'])
			# val2 = trim.sub('', e2['price'])
			val1 = float(e1['price'])
			val2 = float(e2['price'])

			if distance >= required_distance and val1 + val2 >= filter_condition:
				count_total = 1
				match_map[e1['name']].append((e1['name'],e2['name'], count_total))
				# match.append((e1['name'],e2['name'], sum_total))

				res_without_count.append((e1['id'], e2['id']))

	for k,v in match_map.items():
		match_map[k] = sorted(v,key=lambda x: x[-1], reverse=True)

	if is_max == True:
		res = extract_top_n(match_map,num_matches)
	else:
		res = extract_bottom_1(match_map)
	return res, res_without_count

def extract_top_n(d,n):
	res = []
	for k,v in d.items():
		res += v[:n]
	# print(res)
	return res

def extract_bottom_1(d):
	res = []	
	for k,v in d.items():
		res.append(v[-1])
	# print(res)
	return res


# def matcher_dup_updated(d1, d2, distance_fn, sampler_fn, required_distance, sample_size=100):

# 	match = []	

# 	for e1 in d1:
# 		# Note that d1 is always the duplicated table
# 		# So the entries of names need to cleaned for the "_number" adjustment during the matching
# 		cleaned_e1 = e1['name'].split("_")[0]
# 		for e2 in sampler_fn(d2, sample_size):
# 			distance = calc_max_weight_edit(str(cleaned_e1).lower(), str(e2['name']).lower(), distance_fn)
# 			if distance >= required_distance:
# 				sum_total = int(e1['age']) + int(e2['age'])
# 				match.append((e1['name'],e2['name'], sum_total))
# 	return match

# Create a look up reference of the data
def create_lookup(d1,d2, col1, col2):
	records_1 = d1.to_records(index=False)
	result_1 = list(records_1)
	records_2 = d2.to_records(index=False)
	result_2 = list(records_2)
	joined_list = result_1 + result_2
	data_lookup = {}
	for (name,age) in joined_list:
		data_lookup[name] = age
	return data_lookup

# filter naive and random sampling matching results, similar to the function of filter of bipartite
# (See Bipartite Matching COUNT for example)

def filter_results(data_lookup, filter_threshold, matching_list):

	filtered_list = []

	for match in matching_list:
		first = match[0].split("_")[0]
		second = match[1]
		val1 = float(data_lookup[first])
		val2 = float(data_lookup[second])

		if val1 <= filter_threshold and val2 <= filter_threshold:
			filtered_list.append((first, second, val1, val2))

	return filtered_list


def all(catalog, k=None):
	return catalog

def random_sample(catalog, k):
	k = min(k,catalog.length)
	return random.sample(list(catalog),k)

# Function used for Case 3.
def matching_with_random_swaps(num_swaps, num_matches, is_max, d1, d2, distance_fn, sampler_fn, required_distance, sample_size=100):

	# match = []
	swap_counter = 0
	match_map = defaultdict(list)
	for e1 in d1:

		for e2 in sampler_fn(d2, sample_size):
			distance = calc_max_weight_edit(str(e1['name']).lower(), str(e2['name']).lower(), distance_fn)
#			print("M1: ", e1['name'], "M2: ", e2['name'],"DIST: ", distance, "REQ:", required_distance)
			if swap_counter > num_swaps:
				if distance >= required_distance:
					sum_total = int(e1['age']) + int(e2['age'])
					match_map[e1['name']].append((e1['name'],e2['name'], sum_total))
					# match.append((e1['name'],e2['name'], sum_total))
			else:
				sum_total = int(e1['age']) + int(e2['age'])
				match_map[e1['name']].append((e1['name'],e2['name'], sum_total))
				swap_counter += 1
	for k,v in match_map.items():
		match_map[k] = sorted(v,key=lambda x: x[-1], reverse=True)

	if is_max == True:
		res = extract_top_n(match_map,num_matches)
	else:
		res = extract_bottom_1(match_map)
	return res
