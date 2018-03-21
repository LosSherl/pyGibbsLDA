#!/usr/bin/python2.7  
# -*- coding: utf-8 -*-

import random
import utils
from strtokenizer import strtokenizer
from dataset import dataset
from dataset import document

MODEL_STATUS_UNKNOWN = 0
MODEL_STATUS_EST = 1
MODEL_STATUS_ESTC = 2
MODEL_STATUS_INF = 3

class model(object):
	def __init__(self):
		self.set_default_values()

	def set_default_values(self):
		self.wordmapfile = "wordmap.txt"
		self.trainlogfile = "trainlog.txt"
		self.tassign_suffix = ".tassign"
		self.theta_suffix = ".theta"
		self.phi_suffix = ".phi"
		self.others_suffix = ".others"
		self.twords_suffix = ".twords"

		self.directory = "./"
		self.dfile = "trndocs.dat"
		self.model_name = "model-final"    
		self.model_status = MODEL_STATUS_UNKNOWN

		self.ptrndata = None
		self.pnewdata = None

		self.id2word = dict()

		self.M = 0
		self.V = 0
		self.K = 100
		alpha = 50.0 / self.K
		self.beta = 0.1
		self.niters = 2000
		self.liter = 0
		self.self.savestep = 200    
		self.twords = 0
		self.withrawstrs = 0

		self.p = None
		self.z = None
		self.nw = None
		self.nd = None
		self.nwsum = None
		self.ndsum = None
		self.theta = None
		self.phi = None

		self.inf_liter = 0
		self.newM = 0
		self.newV = 0
		self.newz = None
		self.newnw = None
		self.newnd = None
		self.newnwsum = None
		self.newndsum = None
		self.newtheta = None
		self.newphi = None

	def parse_args(self,argc,argv):
		utils.parse_args(argc,argv,self)

	def init(self,argc,argv):
		if self.parse_args(argc,argv):
			return 1

		if model_status == MODEL_STATUS_EST:
			if init_est(): 
				return 1
		elif model_status == MODEL_STATUS_ESTC:
			if init_estc():
				return 1
		elif model_status == MODEL_STATUS_INF:
			if init_inf():
				return 1

		return 0

	def load_model(self,model_name):

		filename = self.directory + model_name + self.tassign_suffix;
		fin = open(filename,"r")

		z = [0] * self.M
		self.ptrndata = dataset(self.M)
		self.ptrndata.V = self.V

		for i in range(M):
			line = fin.readline()
			strtok = strtokenizer(line)
			length = strtok.count_tokens()
			words = []
			topics = []
			for j in range(length):
				token = strtok.token(j)
				tok = strtokenizer(token,":")
				if tok.count_tokens() != 2:
					print "Invalid word-topic assignment line!\n"
					return 1
				words.append(int(tok.token(0)))
				topics.append(int(tok.token(1)))
		
			pdoc = document(len(words),words)
			self.ptrndata.add_doc(pdoc,i)

			self.z[i] = [0] * len(topics)
			for j in range(len(topics)):
				self.z[i][j] = topics[j]

		fin.close()

		return 0

	def save_model(self,model_name):
		if self.save_model_tassign(self.directory + model_name + self.tassign_suffix):
			return 1

		if self.save_model_others(self.directory + model_name + self.others_suffix):
			return 1

		if self.save_model_theta(self.directory + model_name + self.theta_suffix):
			return 1

		if self.save_model_phi(self.directory + model_name + self.phi_suffix):
			return 1

		if self.twords > 0:
			if self.save_model_twords(self.directory + model_name + self.twords_suffix):
				return 1

		return 0

	def save_model_tassign(self,filename):
		fout = open(filename,"w")

		for i in range(self.ptrndata.M):
			for j in range(self.ptrndata.docs[i].length):
				fout.write("%d:%d " % (self.ptrndata.docs[i].words[j],self.z[i][j]))
			fout.write("\n")

		fout.close()
		return 0

	def save_model_theta(self,filename):
		fout = open(filename,"w")
		
		for i in range(self.M):
			for j in range(self.K):
				fout.write("%f ",self.theta[i][j])
			fout.write("\n")
		
		return 0

	def save_model_phi(self,filename):
		fout = open(filename,"w")
		
		for i in range(self.K):
			for j in range(self.V):
				fout.write("%f ",self.phi[i][j])
			fout.write("\n")
		
		return 0

	def save_model_others(self,filename):
		fout = open(filename,"w")

		fout.write("alpha=%f\n" % self.alpha)
		fout.write("beta=%f\n" % self.beta)
		fout.write("ntopics=%d\n" % self.K)
		fout.write("ndocs=%d\n" % self.M)
		fout.write("nwords=%d\n" % self.V)
		fout.write("liter=%d\n" % self.liter)

		fout.close()

		return 0

	def save_model_twords(self,filename):
		fout = open(filename,"w")
		if self.twords > self.V:
			self.twords = self.V
    	
    	for k in range(self.K):
    		words_probs = []
    		for w in range(self.V):
    			word_prob = (w,self.phi[k][w])
    			words_probs.append(word_prob)

    		utils.quicksort(words_probs,0,len(words_probs) - 1)

    		fout.write("Topic %dth:\n" % k)

    		for i in range(self.twords):
    			if self.id2word.get(words_probs[i][0]):
    				fout.write("\t%s   %f\n" % (self.id2word[words_probs[i][0]],words_probs[i][1]))
    
        fout.close()
 
    	return 0

    def save_inf_model(self,model_name):
    	if self.save_inf_model_tassign(self.directory + model_name + self.tassign_suffix):
			return 1

		if self.save_inf_model_others(self.directory + model_name + self.others_suffix):
			return 1

		if self.save_inf_model_newtheta(self.directory + model_name + self.theta_suffix):
			return 1

		if self.save_inf_model_newphi(self.directory + model_name + self.phi_suffix):
			return 1

		if self.twords > 0:
			if self.save_inf_model_twords(self.directory + model_name + self.twords_suffix):
				return 1

		return 0

	def save_inf_model_tassign(self,filename):
		fout = open(filename,"w")

		for i in range(self.pnewdata.M):
			for j in range(self.pnewdata.docs[i].length):
				fout.write("%d:%d " % (self.pnewdata.docs[i].words[j],self.newz[i][j]))
			fout.write("\n")

		fout.close()
		return 0

	def save_inf_model_newtheta(self,filename):
		fout = open(filename,"w")
		
		for i in range(self.newM):
			for j in range(self.K):
				fout.write("%f ",self.newtheta[i][j])
			fout.write("\n")
		
		return 0

	def save_inf_model_newphi(self,filename):
		fout = open(filename,"w")
		
		for i in range(self.K):
			for j in range(self.newV):
				fout.write("%f ",self.newphi[i][j])
			fout.write("\n")
		
		return 0

	def save_inf_model_others(self,filename):
		fout = open(filename,"w")

		fout.write("alpha=%f\n" % self.alpha)
		fout.write("beta=%f\n" % self.beta)
		fout.write("ntopics=%d\n" % self.K)
		fout.write("ndocs=%d\n" % self.newM)
		fout.write("nwords=%d\n" % self.newV)
		fout.write("liter=%d\n" % self.inf_liter)

		fout.close()

		return 0

	def save_inf_model_twords(self,filename):
		fout = open(filename,"w")
		if self.twords > self.newV:
			self.twords = self.newV
    	
    	for k in range(self.K):
    		words_probs = []
    		for w in range(self.newV):
    			word_prob = (w,self.newphi[k][w])
    			words_probs.append(word_prob)

    		utils.quicksort(words_probs,0,len(words_probs) - 1)

    		fout.write("Topic %dth:\n" % k)
    		
    		for i in range(self.twords):
    			if pnewdata._id2id.get(words_probs[i][0]):
    				_it = pnewdata._id2id[words_probs[i][0]]
    				if self.id2word.get(_it):
    					fout.write("\t%s   %f\n" % (self.id2word[_it],words_probs[i][1]))
        
        fout.close()
 
    	return 0

    def init_est(self):
    	p = [0] * self.K

    	self.ptrndata = dataset()
    	if self.ptrndata.read_trndata(self.directory + self.dfile + self.wordmapfile):
    		print "Fail to read training data!\n"
    		return 1

  		self.M = self.ptrndata.M
  		self.V = self.ptrndata.V

		nw = [0] * self.V

		for w in range(self.V):
			nw[w] = [0] * self.K
			for k in range(self.K):
				nw[w][k] = 0

    	nd = [0] * self.M

    	for m in range(self.M):
    		nd[m] = [0] * self.K
    		for k in range(self.K):
    			nd[m][k] = 0
	
	    nwsum = [0] * self.K

	    for k in range(self.K):
	    	nwsum[k] = 0
	
   		ndsum = [0] * M

   		for m in range(self.M):
   			ndsum[m] = 0

   		z = [0] * self.M

   		for m in range(self.ptrndata.M):
   			N = self.ptrndata.docs[m].length()
   			z[m] = [0] * N

   			for n in range(N):
   				topic = int(random.random() * self.K)
   				z[m][n] = topic

   				# word i 被赋予topic j的次数	
   				nw[self.ptrndata.docs[m].words[n]][topic] += 1
   				# document i 中的单词被赋予topic j的次数
   				nd[m][topic] += 1
   				# 被赋予topic j的单词总数
   				nwsum[topic] += 1

   			# document i的单词总数
   			ndsum[m] = N
	
        self.theta = [0] * self.M
        for m in range(self.M):
        	self.theta[m] = [0] * self.K 
   		
   		self.phi = [0] * self.K
   		for k in range(self.K):
   			self.phi[k] = [0] * self.V
	
    
  		return 0;