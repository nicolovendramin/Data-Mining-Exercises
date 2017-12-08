import numpy as np
from scipy.sparse import csc_matrix, csr_matrix
from progressbar import ProgressBar
import argparse
import math
import time
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--k_shingle', type=int, default=10)
parser.add_argument('--signature_length', type=int, default=10)
parser.add_argument('--word_shingling', type=bool, default=False)
parser.add_argument('--threshold', type=float, default=0.5)
parser.add_argument('--documents_directory', type=str, default="data")
args = parser.parse_args()

class Shingling:

	def __init__(self, shingling_size=10):
		self.shingling_size = shingling_size

	def shingle(self, document, k=None, shingle_by_word=False):

		if k==None:
			k = self.shingling_size

		shingles = []
		if shingle_by_word:
			document = document.split(" ")

		for i in range(0,len(document)-k+1):
			shingles.append(document[i:i+k])

		return list(set(shingles))

	@staticmethod
	def shingling(document, k, shingle_by_word=False):

		shingles = []

		if shingle_by_word:
			document = document.split(" ")

		for i in range(0,len(document)-k+1):
			shingles.append(document[i:i+k])

		return list(set(shingles))


class CompareSets:

	@staticmethod
	def JaccardSimilarity(one, two):
		set_one = set(one)
		set_two = set(two)
		common_elements = len(set.intersection(*[set_one, set_two]))
		total_elements = len(set.union(*[set_one, set_two]))

		return float(common_elements) / total_elements


class MinHashing:

	def __init__(self, tot_shingles, n=5):
		self.n = n
		self.a = []
		self.b = []
		self.c = Primes.firstAfter(tot_shingles)
		for i in range(0, n):
			self.a.append(np.random.randint(1, tot_shingles))
			self.b.append(np.random.randint(1, tot_shingles))
	
	def minHash(self, document):
		signature = []
		for i in range(0, self.n):
			min_ = self.c + 1

			for shingle in document:
				val = (self.a[i] * shingle + self.b[i]) % self.c
				if(val<min_):
					min_ = val

			signature.append(min_)

		return signature

	def compareSignatures(self, signature1, signature2):
		common_elements = len(set.intersection(*[set(signature1), set(signature2)]))

		return float(common_elements)/self.n


class Primes:

	@staticmethod
	def isPrime(num):
		for i in range(2, num/2 + 1):
			if float(num)/i - int(num/i) == 0:
				return False
		return True

	@staticmethod
	def firstAfter(num):
		while(True):
			if Primes.isPrime(num):
				return num
			else:
				num = num + 1


class LocalitySensitiveHashing:

	def __init__(self, bands=5, buckets_num=29):
		self.bands = bands
		self.buckets_num = Primes.firstAfter(buckets_num)
		self.a = np.random.randint(1, buckets_num)
		self.b = np.random.randint(1, buckets_num)
		self.buckets = [[[] for i in range(0, self.buckets_num)] for i in range(0, self.bands)]

	def lsHashing(self, signature, signature_id):
		step = int(len(signature)/self.bands)
		partial_sum = 0
		band = 0

		for i in range(1, len(signature)+1):
			partial_sum += signature[i-1]
			if i%step == 0:
				self.buckets[band][(self.a*partial_sum + self.b) % self.buckets_num].append(signature_id)
				partial_sum = 0
				band += 1

		if i%step!=0:
			self.buckets[(self.a*partial_sum + self.b) % self.buckets_num].append(signature_id)

	def getPossibleMatches(self):
		possible_matches = []
		for band in self.buckets:
			for bucket in band:
				for i in range(0, len(bucket)-1):
					for j in range(i+1, len(bucket)):
						possible_matches.append((bucket[i], bucket[j]))

		possible_matches = list(set(possible_matches))

		return possible_matches

	@staticmethod
	def localSensitivityHashing(signatures_matrix, bands, buckets_num):
		a = np.random.randint(1, buckets_num)
		b = np.random.randint(1, buckets_num)

		similar_item_ids = []
		number_of_documents = signatures_matrix.shape[0]
		signature_length = signatures_matrix.shape[1]

		step = signature_length / bands

		for band in range(bands):
			buckets = []
			for document in range(0, number_of_documents):
				value = 0
				for signature in range(band, band + step):
					value = value + (signatures_matrix[document][signature]*a + b)
				
				hashed_value = value % buckets_num
				buckets.append(hashed_value)

			for candidate_one in range(0, len(buckets)):
				for candidate_two in range(candidate_one+1, len(buckets)):
					if(buckets[candidate_one] == buckets[candidate_two]):
						similar_item_ids.append((candidate_one, candidate_two))
		
		return list(set(similar_item_ids))


def main(file_path, num):
	threshold = args.threshold

	# Importing the dataset

	documents = []

	f = open(file_path, "rU")
	for i in range(0, num): 
	  words = f.readline()
	  documents.append(words[words.find(" ")+1:])
	f.close()
	
	# SHINGLING PHASE
	print("Shingling Phase:")
	# I backup the textual version of the dataset before starting to transform it	
	docs = list(documents)
	shingling_time = time.time()

	# For each document i map it into the set of its shingles
	documents = [Shingling.shingling(document, args.k_shingle, args.word_shingling) for document in documents]

	# I unify all the shingles to have the set of all the shingles present in all the analysed documents
	shingles = list(set([shingle for document in documents for shingle in document]))

	# I build a dictionary mapping each shingle to an integer value
	shingle_dic = {shingles[i]: i for i in range(0, len(shingles))}

	size = len(documents)
	shingle_doc_pairs = []
	rows = []
	cols = []
	data = []

	# I generate the auxiliary data structures to build the sparse representation of the characteristic matrix
	for document in range(0, len(documents)):
		for shingle in documents[document]:
			data.append(1)
			# each row represents a shingleId
			rows.append(shingle_dic[shingle])
			# each columns correspond to a document
			cols.append(document)
	
	# I transform each document from list of shingles to list of shingleIds
	documents = [[shingle_dic[shingle] for shingle in document] for document in documents]

	# Initialization of the characteristic document_matrix
	document_matrix = csc_matrix((data, (rows, cols)))

	shingling_time = time.time() - shingling_time
	print("Shingling completed.")
 
	# I instatntiate a minHasher and an empty list to contain the signatures
	minHasher = MinHashing(len(shingles), args.signature_length)
	signatures = []

	# I start the minHashing procedure	
	print("MinHashing Phase:")
	min_hashing_time = time.time()
	pbar = ProgressBar()

	# Iterate on the columns of the characteristic matrix (each column corresponds to a document)
	for i in pbar(range(0, len(document_matrix.indptr)-1)):
		# Defining the limits of the iteration on the indices 	
		min_ = document_matrix.indptr[i]
		max_ = document_matrix.indptr[i+1]
		# I append to the signatures the signature of the document as a minHash of the shinglesId it contains
		signatures.append(minHasher.minHash(document_matrix.indices[min_:max_]))

	# I build the signature matrix as an numpy matrix
	signatures = np.asarray([np.asarray(signature) for signature in signatures])

	min_hashing_time = time.time() - min_hashing_time

	similar_docs = []

	
	lshasher = LocalitySensitiveHashing(buckets_num=Primes.firstAfter(len(shingles)))

	print("LSHashing Phase:")

	lsh_comparison_time = time.time()
	
	sig_size = len(signatures)
	for i in range(0, sig_size):
		lshasher.lsHashing(signatures[i], i)

	to_test = lshasher.getPossibleMatches()

	for candidate_one, candidate_two in to_test:
		minHashSimilarity = minHasher.compareSignatures(signatures[candidate_one], signatures[candidate_two])
		if minHashSimilarity >= threshold:
			jack = CompareSets.JaccardSimilarity(documents[candidate_one], documents[candidate_two])
			similar_docs.append((candidate_one, candidate_two, minHashSimilarity, jack))

	print("LSHashing completed.")
	lsh_comparison_time = time.time() - lsh_comparison_time

	print("Evaluation phase:")
	avg_diff = 0
	std_deviation = 0
	
	if len(similar_docs) > 0:

		for similar_doc in similar_docs:
			avg_diff += math.fabs(similar_doc[3] - similar_doc[2])

		avg_diff = avg_diff / len(similar_docs)

		for similar_doc in similar_docs:
			std_deviation += math.pow(math.fabs(similar_doc[3] - similar_doc[2]) - avg_diff, 2)

		std_deviation = std_deviation / len(similar_docs)

	print("Evaluation completed.")

	print("RESULTS:{} similarity pairs,with {} average difference ({} std. deviation) between estimated and jaccard similarities"
		).format(len(similar_docs), avg_diff, std_deviation)

	return  num, len(similar_docs), avg_diff, std_deviation, shingling_time, min_hashing_time, lsh_comparison_time


# Import os to navigate the directories
import os 

topic_dir = args.documents_directory
all_files = os.listdir(topic_dir)[1:]
print(all_files)
results = []
for path in all_files:
	num = int(path.replace("articles_", "").replace(".train", ""))
	print("Testing with " + str(num) + " articles.")
	final_path = topic_dir + "/" + path
	results.append(main(final_path, num))

data = results

