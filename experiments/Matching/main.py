import editdistance
import core
import analyze
import matcher
import sys

sample_size = 1000

print('Loading catalogs...')
amzn = core.amazon_catalog()
goog = core.google_catalog()

print('Performing compare all match (edit distance)...')
compare_all_edit_match = matcher.matcher(amzn,goog,editdistance.eval, matcher.all)
print('Compare All Matcher (Edit Distance) Performance: ' + str(core.eval_matching(compare_all_edit_match)))

print('Performing compare all match (jaccard distance)...')
compare_all_jaccard_match = matcher.matcher(amzn,goog,analyze.jaccard_calc, matcher.all)
print('Compare All Matcher (Jaccard Distance) Performance: ' + str(core.eval_matching(compare_all_jaccard_match)))

print('Performing random sample match (edit distance)...')
compare_all_edit_match = matcher.matcher(amzn,goog,editdistance.eval, matcher.random_sample, sample_size)
print('Random Sample Matcher (Edit Distance) Performance: ' + str(core.eval_matching(compare_all_edit_match)))

print('Performing random sample match (jaccard distance)...')
compare_all_jaccard_match = matcher.matcher(amzn,goog,analyze.jaccard_calc, matcher.random_sample, sample_size)
print('Random Sample Matcher (Jaccard Distance) Performance: ' + str(core.eval_matching(compare_all_jaccard_match)))