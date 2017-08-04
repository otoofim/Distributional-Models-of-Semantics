# coding: utf-8

import gensim
import math
from copy import copy
from math import*

vocabsize=5000

'''
(f) helper class, do not modify.
provides an iterator over sentences in the provided BNC corpus
input: corpus path to the BNC corpus
input: n, number of sentences to retrieve (optional, standard -1: all)
'''
class BncSentences:
	def __init__(self, corpus, n=-1):
		self.corpus = corpus
		self.n = n
	
	def __iter__(self):
		n = self.n
		ret = []
		for line in open(self.corpus):
			line = line.strip().lower()
			if line.startswith("<s "):
				ret = []
			elif line.strip() == "</s>":
				if n > 0:
					n -= 1
				if n == 0:
					break
				yield copy(ret)
			else:
				parts = line.split("\t")
				if len(parts) == 3:
					word = parts[-1]
					idx = word.rfind("-")
					word, pos = word[:idx], word[idx+1:]
					if word in ['thus', 'late', 'often', 'only', 'usually', 'however', 'lately', 'absolutely', 'hardly', 'fairly', 'near', 'similarly', 'sooner', 'there', 'seriously', 'consequently', 'recently', 'across', 'softly', 'together', 'obviously', 'slightly', 'instantly', 'well', 'therefore', 'solely', 'intimately', 'correctly', 'roughly', 'truly', 'briefly', 'clearly', 'effectively', 'sometimes', 'everywhere', 'somewhat', 'behind', 'heavily', 'indeed', 'sufficiently', 'abruptly', 'narrowly', 'frequently', 'lightly', 'likewise', 'utterly', 'now', 'previously', 'barely', 'seemingly', 'along', 'equally', 'so', 'below', 'apart', 'rather', 'already', 'underneath', 'currently', 'here', 'quite', 'regularly', 'elsewhere', 'today', 'still', 'continuously', 'yet', 'virtually', 'of', 'exclusively', 'right', 'forward', 'properly', 'instead', 'this', 'immediately', 'nowadays', 'around', 'perfectly', 'reasonably', 'much', 'nevertheless', 'intently', 'forth', 'significantly', 'merely', 'repeatedly', 'soon', 'closely', 'shortly', 'accordingly', 'badly', 'formerly', 'alternatively', 'hard', 'hence', 'nearly', 'honestly', 'wholly', 'commonly', 'completely', 'perhaps', 'carefully', 'possibly', 'quietly', 'out', 'really', 'close', 'strongly', 'fiercely', 'strictly', 'jointly', 'earlier', 'round', 'as', 'definitely', 'purely', 'little', 'initially', 'ahead', 'occasionally', 'totally', 'severely', 'maybe', 'evidently', 'before', 'later', 'apparently', 'actually', 'onwards', 'almost', 'tightly', 'practically', 'extremely', 'just', 'accurately', 'entirely', 'faintly', 'away', 'since', 'genuinely', 'neatly', 'directly', 'potentially', 'presently', 'approximately', 'very', 'forwards', 'aside', 'that', 'hitherto', 'beforehand', 'fully', 'firmly', 'generally', 'altogether', 'gently', 'about', 'exceptionally', 'exactly', 'straight', 'on', 'off', 'ever', 'also', 'sharply', 'violently', 'undoubtedly', 'more', 'over', 'quickly', 'plainly', 'necessarily']:
						pos = "r"
					if pos == "j":
						pos = "a"
					ret.append(gensim.utils.any2unicode(word + "." + pos))
'''
This is a help function to create a full array from a sparse one
'''
def convert_sparse_to_full(vector1):
	temp_vector1 = [0] * vocabsize
	for indx, value in vector1:
		temp_vector1[int(indx)] = int(value)

	return temp_vector1

'''
Sometimes even full arrays don't have correct dimentioin.
So, with this help function they are made ready for further proccessing.
'''
def convert_full_to_full(vector1):
	temp_vector1 = [0] * vocabsize

	for indx, value in enumerate(vector1):
		temp_vector1[int(indx)] = int(value)

	return temp_vector1



def square_rooted(x):
	return sqrt(sum([a * a for a in x]))

'''
(a) function load_corpus to read a corpus from disk
input: vocabFile containing vocabulary
input: contextFile containing word contexts
output: id2word mapping word IDs to words
output: word2id mapping words to word IDs
output: vectors for the corpus, as a list of sparse vectors
'''
def load_corpus(vocabFile, contextFile):
	id2word = {}
	word2id = {}
	vectors = []
	qq=0
	# your code here
	with open(vocabFile) as v, open (contextFile) as c:
		for idx,line in enumerate(v):
			word2id[line.strip()] = int(idx)
			id2word[int(idx)] = line.strip()
		for lines in c:
			qq= qq+1
			tokens = lines.strip().split()
			temp = [0] * vocabsize
			for counts in  tokens[1:]:
				indx, cou = counts.split(":")
				temp[int(indx)] = cou
			vectors.append(temp)
	return id2word, word2id, vectors

'''
(b) function cosine_similarity to calculate similarity between 2 vectors
input: vector1
input: vector2
output: cosine similarity between vector1 and vector2 as a real number
'''
def cosine_similarity(vector1, vector2):
	# your code here
	temp1=[]
	temp2=[]

	if not isinstance(vector1[0], tuple):
		temp1 = convert_full_to_full(vector1)
	else:
		temp1 = convert_sparse_to_full(vector1)

	if not isinstance(vector2[0], tuple):
		temp2 = convert_full_to_full(vector2)
	else:
		temp2 = convert_sparse_to_full(vector2)

	numerator = sum(a * b for a, b in zip(temp1, temp2))
	denominator = square_rooted(temp1) * square_rooted(temp2)

	return (numerator / denominator)

'''
(d) function tf_idf to turn existing frequency-based vector model into tf-idf-based vector model
input: freqVectors, a list of frequency-based vectors
output: tfIdfVectors, a list of tf-idf-based vectors
'''
def tf_idf(freqVectors):
	tfIdfVectors = []
	
	# your code here
	freqVectorsFull = []
	dfi = [0] * vocabsize
	for vectors in freqVectors:

		fullVector = convert_full_to_full(vectors)

		freqVectorsFull.append(fullVector)
		for worindx, freq in enumerate(fullVector):
			if freq > 0:
				dfi[worindx] += 1
	N = len(freqVectors)
	for vectors in freqVectorsFull:
		temp = []
		for indx, freq in enumerate(vectors):
			if freq > 0:
				temp.append((1 + math.log(freq, 2)) * (1 + math.log((N / dfi[indx]), 2)))
			else:
				temp.append(0)
		tfIdfVectors.append(temp)


	return tfIdfVectors

'''
(f) function word2vec to build a word2vec vector model with 100 dimensions and window size 5
'''
def word2vec(corpus, learningRate, downsampleRate, negSampling):
	# your code here
	return None

'''
(h) function lda to build an LDA model with 100 topics from a frequency vector space
input: vectors
input: wordMapping mapping from word IDs to words
output: an LDA topic model with 100 topics, using the frequency vectors
'''
def lda(vectors, wordMapping):
	# your code here
	return None

'''
(j) function get_topic_words, to get words in a given LDA topic
input: ldaModel, pre-trained Gensim LDA model
input: topicID, ID of the topic for which to get topic words
input: wordMapping, mapping from words to IDs (optional)
'''
def get_topic_words(ldaModel, topicID):
	# your code here
	return None

if __name__ == '__main__':
	import sys
	
	part = sys.argv[1].lower()
	
	# these are indices for house, home and time in the data. Don't change.
	house_noun = 80
	home_noun = 143
	time_noun = 12
	
	# this can give you an indication whether part a (loading a corpus) works.
	# not guaranteed that everything works.
	if part == "a":
		print("(a): load corpus")
		try:
			id2word, word2id, vectors = load_corpus(sys.argv[2], sys.argv[3])
			if not id2word:
				print("\tError: id2word is None or empty")
				exit()
			if not word2id:
				print("\tError: id2word is None or empty")
				exit()
			if not vectors:
				print("\tError: id2word is None or empty")
				exit()
			print("\tPass: load corpus from file")
		except Exception as e:
			print("\tError: could not load corpus from disk")
			print(e)
		
		try:
			if not id2word[house_noun] == "house.n" or not id2word[home_noun] == "home.n" or not id2word[time_noun] == "time.n":
				print("\tError: id2word fails to retrive correct words for ids")
			else:
				print("\tPass: id2word")
		except Exception:
			print("\tError: Exception in id2word")
			print(e)
		
		try:
			if not word2id["house.n"] == house_noun or not word2id["home.n"] == home_noun or not word2id["time.n"] == time_noun:
				print("\tError: word2id fails to retrive correct ids for words")
			else:
				print("\tPass: word2id")
		except Exception:
			print("\tError: Exception in word2id")
			print(e)
	
	# this can give you an indication whether part b (cosine similarity) works.
	# these are very simple dummy vectors, no guarantee it works for our actual vectors.
	if part == "b":
		import numpy
		print("(b): cosine similarity")
		try:
			cos = cosine_similarity([(0,1), (2,1), (4,2)], [(0,1), (1,2), (4,1)])
			if not numpy.isclose(0.5, cos):
				print("\tError: sparse expected similarity is 0.5, was {0}".format(cos))
			else:
				print("\tPass: sparse vector similarity")
		except Exception:
			print("\tError: failed for sparse vector")
		try:
			cos = cosine_similarity([1, 0, 1, 0, 2], [1, 2, 0, 0, 1])
			if not numpy.isclose(0.5, cos):
				print("\tError: full expected similarity is 0.5, was {0}".format(cos))
			else:
				print("\tPass: full vector similarity")
		except Exception:
			print("\tError: failed for full vector")

	# you may complete this part to get answers for part c (similarity in frequency space)
	if part == "c":
		# your code here

		print("(c) similarity of house, home and time in frequency space")
		id2word, word2id, vectors = load_corpus(sys.argv[2], sys.argv[3])
		house_id = word2id["house.n"]
		home_id = word2id["home.n"]
		time_id = word2id["time.n"]
		print("The similarity between 'house' and 'home' is : {0}".format(cosine_similarity(vectors[house_id],vectors[home_id])))
		print("The similarity between 'house' and 'time' is : {0}".format(cosine_similarity(vectors[house_id], vectors[time_id])))
		print("The similarity between 'home' and 'time' is : {0}".format(cosine_similarity(vectors[home_id], vectors[time_id])))


	# this gives you an indication whether your conversion into tf-idf space works.
	# this does not test for vector values in tf-idf space, hence can't tell you whether tf-idf has been implemented correctly
	if part == "d":
		print("(d) converting to tf-idf space")
		id2word, word2id, vectors = load_corpus(sys.argv[2], sys.argv[3])
		try:
			tfIdfSpace = tf_idf(vectors)
			if not len(vectors) == len(tfIdfSpace):
				print("\tError: tf-idf space does not correspond to original vector space")
			else:
				print("\tPass: converted to tf-idf space")
		except Exception as e:
			print("\tError: could not convert to tf-idf space")
			print(e)
	
	# you may complete this part to get answers for part e (similarity in tf-idf space)
	if part == "e":
		print("(e) similarity of house, home and time in tf-idf space")
		
		# your code here
		id2word, word2id, vectors = load_corpus(sys.argv[2], sys.argv[3])
		tfIdfSpace = tf_idf(vectors)
		house_id = word2id["house.n"]
		home_id = word2id["home.n"]
		time_id = word2id["time.n"]
		print("The similarity between 'house' and 'home' is : {0}".format(cosine_similarity(tfIdfSpace[house_id], tfIdfSpace[home_id])))
		print("The similarity between 'house' and 'time' is : {0}".format(cosine_similarity(tfIdfSpace[house_id], tfIdfSpace[time_id])))
		print("The similarity between 'home' and 'time' is : {0}".format(cosine_similarity(tfIdfSpace[home_id], tfIdfSpace[time_id])))

	# you may complete this part for the first part of f (estimating best learning rate, sample rate and negative samplings)
	if part == "f1":
		print("(f1) word2vec, estimating best learning rate, sample rate, negative sampling")
		
		# your code here
	
	# you may complete this part for the second part of f (training and saving the actual word2vec model)
	if part == "f2":
		import logging
		logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
		print("(f2) word2vec, building full model with best parameters. May take a while.")
		
		# your code here
	
	# you may complete this part to get answers for part g (similarity in your word2vec model)
	if part == "g":
		print("(g): word2vec based similarity")
		
		# your code here
	
	# you may complete this for part h (training and saving the LDA model)
	if part == "h":
		import logging
		logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
		print("(h) LDA model")
		
		# your code here
	
	# you may complete this part to get answers for part i (similarity in your LDA model)
	if part == "i":
		print("(i): lda-based similarity")
		
		# your code here

	# you may complete this part to get answers for part j (topic words in your LDA model)
	if part == "j":
		print("(j) get topics from LDA model")
		
		# your code here
