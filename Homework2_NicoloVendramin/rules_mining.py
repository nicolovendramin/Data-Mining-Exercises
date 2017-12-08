
import argparse
import math
import time
import itertools

parser = argparse.ArgumentParser()
parser.add_argument('--support', type=int, default=8)
parser.add_argument('--confidence', type=float, default=0.5)
parser.add_argument('--interest', type=float, default=0.3)
parser.add_argument('--rules', type=str, default="True")
parser.add_argument('--verbose', type=str, default="True")
parser.add_argument('--file_path', type=str, default="T10I4D100K.dat.txt")
args = parser.parse_args()


def contains(itemSet, basket, singletons=False):
	"""
	This function defines wether itemSet is contained in basket. If singleton is true
	it means that itemSet only contains one element.
	"""
	if singletons:
		return itemSet in set(basket)

	else:
		if len(itemSet) > len(basket):
			return False

		return set(itemSet) <= set(basket)


def count_support(itemSets, baskets, singletons=False):
	"""
	This function verifies the support of all the itemSet in itemSets in the baskets.
	If singleton is true it means that itemSet is composed by a single value.
	"""
	itemSetsLen = len(itemSets)
	counts = [0 for i in range(0, itemSetsLen)]

	for itemSetIndex in range(0, itemSetsLen):
		itemSet = itemSets[itemSetIndex]

		for basket in baskets:
			if contains(itemSet, basket, singletons):
				counts[itemSetIndex] += 1

	return counts, itemSetsLen


def filter_by_support(itemSets, baskets, threshold=0, singletons=False):
	"""
	Filters the itemSet in itemSets according to the given threshold of miniumum 
	required support.
	"""
	supports, length = count_support(itemSets, baskets, singletons)
	filtered = []
	counts = []

	for i in range(0, length):
		sup = supports[i]
		if sup >= threshold:
			filtered.append(itemSets[i])
			counts.append(sup)

	return filtered, counts, len(counts)


def map_all_items(baskets):
	"""
	This function extracts the singletons and maps them in a dictionary.
	"""
	uniqueItems = set()
	for basket in baskets:
		uniqueItems.update(basket)

	uniqueItems = list(uniqueItems)
	lenUniqueItems = len(uniqueItems)
	uniqueItemsDictionary = {uniqueItems[i]: i for i in range(0, lenUniqueItems)}

	return uniqueItems, uniqueItemsDictionary, lenUniqueItems


def apriori(baskets, threshold=1):
	"""
	Implementation of the apriori algorithm to find the frequent itemsets given a certain 
	support threshold.
	"""
	uniqueItems, uniqueItemsDictionary, lenUniqueItems = map_all_items(baskets)
	filtered_singletons, supports, lenSingletons = filter_by_support(uniqueItems, baskets, threshold, singletons=True)

	condition = True
	itemSets = []
	itemSets.append([[i] for i in filtered_singletons])
	support_dictionary = {}

	for i in range(0, lenSingletons):
		support_dictionary[str(itemSets[0][i])] = supports[i]

	k = 1

	while(True):
		candidates = generate_candidates(itemSets[k-1], filtered_singletons, k)
		num__ = len(candidates)
		candidates, candidate_support, num_ = filter_by_support(candidates, baskets, threshold)

		print("Generated {} candidates, {} remaining after the support filter.".format(num__, num_))

		if num_ > 0:
			itemSets.append(candidates)

			for i in range(0, num_):
				support_dictionary[str(candidates[i])] = candidate_support[i]

			k += 1
			print("Expanding the set with {} ItemSets of {} elements.".format(num_, k))

		else:
			print("Reached maximum expansion of ItemSets.")
			break

	return itemSets, support_dictionary


def generate_candidates(itemSets, singletons, size):
	"""
	This function is used in the apriori algorithm to generate new possible itemSets starting from
	the ones of the previous step and the filtered singletons of the dataset.
	"""
	candidates = []
	singletons_len = len(singletons)
	
	print("Generating candidates of size {}.".format(size+1))
	if size == 1:
		for i in range(0, singletons_len-1):
			for j in range(i+1, singletons_len):
				candidates.append([singletons[i], singletons[j]])
	else:
		for itemSet in itemSets:
			max_ = max(itemSet)
			for singleton in singletons:
				if singleton > max_:
					new = list(itemSet)
					new.append(singleton)
					flag = True
					for subset in itertools.combinations(new, size):
						if list(subset) not in itemSets:
							flag = False
							break

					if flag:
						candidates.append(new)

	return candidates


def generate_rules(frequentItemSets, supports, baskets_len, confidenceThreshold=0.5, interestThreshold=0.3):
	"""
	This function returns the rules with a certain confidence and the the ones with a certain interest 
	finding them as subsets of frequent itemsets.
	"""
	max_set_size = len(frequentItemSets)

	rules = []
	interesting_rules = []

	for size in range(1, max_set_size):
		itemSets = frequentItemSets[size]
		for itemSet in itemSets:
			itemSet_support = supports[str(itemSet)]
			for item in itemSet:
				set_ = list(itemSet)
				set_.remove(item)
				rule = [set_, item]
				confidence = itemSet_support / supports[str(set_)]
				if confidence >= confidenceThreshold:
					interest = math.fabs(confidence - (supports[str([item])] / baskets_len))
					cand = (rule, confidence, interest, itemSet_support)
					rules.append(cand)
					if interest >= interestThreshold:
						interesting_rules.append(cand)

	return rules, interesting_rules


def discover_association_rules(baskets, supportThreshold=1, confidenceThreshold=0.5, interestThreshold=0.3):
	"""
	This function simply pipelines the function calls to directly find rules.
	"""
	itemSets, support_dictionary = apriori(baskets, supportThreshold)
	return generate_rules(itemSets, support_dictionary, len(baskets), confidenceThreshold, interestThreshold)


def print_rule(rule_tuple):
	"""
	Just a function to print in a formatted style the rule tuple
	"""
	print("[{} -> {}], with confidence of {}, interest of {} and support of {}."
		.format(rule_tuple[0][0], rule_tuple[0][1], rule_tuple[1], rule_tuple[2], rule_tuple[3]))


def print_itemset(itemSets, supports):
	"""
	Just a function to print in a formatted style the rule tuple
	"""
	for layer in itemSets:
		for itemSet in layer:
			print("[{}], with support of {}."
			.format(itemSet, supports[str(itemSet)]))


def main():

	f = open(args.file_path, "rU")
	words = f.readline()
	baskets = []
	while(len(words)>0): 
		baskets.append([int(word) for word in words.split(" ")[:-1]])
		words = f.readline()
	f.close()
	baskets_len = len(baskets)

	if args.rules == "True":
		rules, interesting_rules = discover_association_rules(baskets, args.support, args.confidence, args.interest)
		print("{} rules found. {} of them are kept after the interest filtering.".format(len(rules), len(interesting_rules)))
		if args.verbose == "True":
			for rule in interesting_rules:
				print_rule(rule)

	else:
		itemSets, supports = apriori(baskets, 2)
		print("{} itemSets found.".format(len(supports)))
		if args.verbose == "True":
			print_itemset(itemSets, supports)



main()