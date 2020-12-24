from Matching import core
import datetime
'''
I've tried MinHash, a mix of Jaccard and MinHash, and only Jaccard. 
However, using only Jaccard similarity gave me 
the best accuracy result I could achieve (45% Accuracy) 
in the most efficient computation time (38.41 Seconds). 
The other options I tried either took more computation time or achieved less accuracy.
'''
def tokenize(st):
    return set(st.lower().split())

def jaccard_calc(amazon_prod, google_prod):
    Tokenized_amazon = tokenize(amazon_prod)
    Tokenized_google = tokenize(google_prod)
    
    return -len(Tokenized_amazon.intersection(Tokenized_google))/len(Tokenized_amazon.union(Tokenized_google))

def match():

	matches = []

	for amazon_prod in amazon_catalog():
		for google_prod in google_catalog():
			if jaccard_calc(amazon_prod['title'],google_prod['title']) >= 0.5:
				matches.append((amazon_prod['id'],google_prod['id']))
	return matches
 	

'''
Match must return a list of tuples of amazon ids and google ids.
For example:
[('b000jz4hqo', http://www.google.com/base/feeds/snippets/11125907881740407428'),....]

'''
# AMAZON --> title
# GOOGLE --> name

#prints out the accuracy
# now = datetime.datetime.now()
# out = eval_matching(match())
# timing = (datetime.datetime.now()-now).total_seconds()
# print("----Accuracy----")
# print(out)
# print("---- Timing ----")
# print(timing,"seconds")