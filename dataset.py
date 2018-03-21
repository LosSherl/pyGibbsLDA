#!/usr/bin/python2.7  
# -*- coding: utf-8 -*-

from strtokenizer import strtokenizer

class document(object):
	def __init__(self,length = 0,words = None,rawstr = ""):
		self.length = length
		self.rawstr = rawstr
		self.words = words
		if length > 0 and words == None:
			self.words = [0] * length 

class dataset(object):
	def __init__(self, M = 0):
		self.M = M
		self.V = 0
		self._id2id = dict()
		self.docs = [0] * M
		self._docs = []

	def add_doc(self,doc,idx):
		if 0 <= idx and idx < self.M:
			self.docs[idx] = doc

	def _add_doc(self,doc,idx):
		if 0 <= idx and idx < self.M:
			self._docs[idx] = doc

	@staticmethod
	def write_wordmap(wordmapfile,pword2id):
		fobj = open(wordmapfile,"w")


		fobj.write("%d\n" % len(pword2id))

		for item in pword2id:
			fobj.write("%s %d\n" % (item,pword2id[item]))

		return 0

	@staticmethod
	def read_word2id(wordmapfile,pword2id):
		pword2id = dict()

		fobj = open(wordmapfile,"r")

		nwords = int(fobj.readline())

		for i in range(nwords):
			buff = fobj.readline()
			strtok = strtokenizer(buff)
			
			if strtok.count_tokens() != 2:
				continue

			pword2id[strtok.token(0)] = strtok.token(1)

	@staticmethod
	def read_id2word(wordmapfile,pid2word):
		pid2word = dict()

		fobj = open(wordmapfile,"r")

		nwords = int(fobj.readline())

		for i in range(nwords):
			buff = fobj.readline()
			strtok = strtokenizer(buff)
			
			if strtok.count_tokens() != 2:
				continue

			pid2word[strtok.token(1)] = strtok.token(0)

	def read_trndata(self,dfile,wordmapfile):
		word2id = dict()

		fobj = open(dfile,"r")

		self.M = int(fobj.readline())

		if self.M <= 0:
			print "No document available!\n"
		
		self.docs = [document()] * self.M
		self.V = 0

		for i in range(M):
			line = fobj.readline()
			strtok = strtokenizer(line)
			length = strtok.count_tokens()

			if length <= 0:
				print "Invalid (empty) document!\n"
				self.M = 0
				self.V = 0
				return 1

			pdoc = document(length)

			for j in range(length):
				if word2id.get(strtok.token(j)):
					pdoc.words[j] = word2id[strtok.token(j)]
				else:
					pdoc.words[j] = len(word2id)
					word2id[strtok.token(j)] = len(word2id)

			self.add_doc(pdoc,i)
	
		fobj.close()

		if dataset.write_wordmap(wordmapfile,word2id):
			return 1
	
		self.V = len(word2id)
	
		return 0

	def read_newdata(self,dfile,wordmapfile):
		word2id = dict()
		id2_id = dict()

		dataset.read_wordmap(wordmapfile,word2id)

		if len(word2id) <= 0:
			print "No word map available!\n"
			return 1

		fobj = open(dfile,"r")

		self.M = int(fobj.readline())

		if self.M <= 0:
			print "No document available!\n"
			return 1

		self.docs = [document()] * self.M
		self._docs = [document()] * self.M
		self.V = 0

		for i in range(M):
			line = fobj.readline()
			strtok = strtokenizer(line)
			length = strtok.count_tokens()
			
			doc = []
			_doc = []

			for j in range(length):
				if word2id.get(strtok.token(j)):
					if id2_id.get(word2id[strtok.token(j)]):
						_id = id2_id[word2id[strtok.token(j)]]
					else:
						_id = len(id2_id)
						id2_id[word2id[strtok.token(j)]] = _id
						self._id2id[_id] = word2id[strtok.token(j)]
				else:
					# word not found, i.e., word unseen in training data
					pass

				doc.append(word2id[strtok.token(j)])
				_doc.append(_id)
			
			pdoc = document(len(doc),doc)
			_pdoc = document(len(_doc),_doc)
 			self.add_doc(pdoc)
 			self._add_doc(_pdoc)

 		fobj.close()
		self.V = len(id2_id)
	
		return 0

	def read_newdata_withrawstrs(self,dfile,wordmapfile):
		word2id = dict()
		id2_id = dict()

		dataset.read_wordmap(wordmapfile,word2id)

		if len(word2id) <= 0:
			print "No word map available!\n"
			return 1
		
		fobj = open(dfile,"r")

		self.M = int(fobj.readline())

		if self.M <= 0:
			print "No document available!\n"
			return 1

		self.docs = [document()] * self.M
		self._docs = [document()] * self.M
    
		self.V = 0

		for i in range(M):
			line = fobj.readline()
			strtok = strtokenizer(line)
			length = strtok.count_tokens()
			
			doc = []
			_doc = []

			for j in range(length - 1):
				if word2id.get(strtok.token(j)):
					if id2_id.get(word2id[strtok.token(j)]):
						_id = id2_id[word2id[strtok.token(j)]]
					else:
						_id = len(id2_id)
						id2_id[word2id[strtok.token(j)]] = _id
						self._id2id[_id] = word2id[strtok.token(j)]
				else:
					# word not found, i.e., word unseen in training data
					pass
				doc.append(word2id[strtok.token(j)])
				_doc.append(_id)

			pdoc = document(len(doc),doc,line)
			_pdoc = document(len(_doc),_doc,line)
			self.add_doc(pdoc)
			self._add_doc(_pdoc)

		fobj.close()
		self.V = len(id2_id)
	
		return 0