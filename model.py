#!/usr/bin/python2.7  
# -*- coding: utf-8 -*-

import random
import numpy as np
import utils
import time
from strtokenizer import strtokenizer
from dataset import dataset
from dataset import document
from datetime import datetime

MODEL_STATUS_UNKNOWN = 0
MODEL_STATUS_EST = 1
MODEL_STATUS_INF = 2

class LDA(object):
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
		self.alpha = 50.0 / self.K
		self.beta = 0.1
		self.niters = 2000
		self.liter = 0
		self.savestep = 200    
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
		self = utils.parse_args(argc,argv,self)
		return True

	def init(self,argc,argv):
		if not self.parse_args(argc,argv):
			return False
		if self.model_status == MODEL_STATUS_EST:
			if not self.init_est(): 
				return False
		elif self.model_status == MODEL_STATUS_INF:
			if not self.init_inf():
				return False

		return True

	def load_model(self,model_name):

		filename = self.directory + model_name + self.tassign_suffix
		fin = open(filename,"r")

		self.z = [0] * self.M
		self.ptrndata = dataset(self.M)
		self.ptrndata.V = self.V

		for i in range(self.M):
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
					return False
				words.append(int(tok.token(0)))
				topics.append(int(tok.token(1)))
		
			pdoc = document(len(words),words)
			self.ptrndata.add_doc(pdoc,i)
			self.z[i] = np.zeros(len(topics))
			for j in range(len(topics)):
				self.z[i][j] = topics[j]

		fin.close()
		
		return True

	def save_model(self,model_name):
		if not self.save_model_tassign(self.directory + model_name + self.tassign_suffix):
			return False

		if not self.save_model_others(self.directory + model_name + self.others_suffix):
			return False

		if not self.save_model_theta(self.directory + model_name + self.theta_suffix):
			return False

		if not self.save_model_phi(self.directory + model_name + self.phi_suffix):
			return False

		if self.twords > 0:
			if not self.save_model_twords(self.directory + model_name + self.twords_suffix):
				return False

		return True

	def save_model_tassign(self,filename):
		fout = open(filename,"w")

		for i in range(self.ptrndata.M):
			for j in range(self.ptrndata.docs[i].length):
				fout.write("%d:%d " % (self.ptrndata.docs[i].words[j],self.z[i][j]))
			fout.write("\n")

		fout.close()
		return True

	def save_model_theta(self,filename):
		fout = open(filename,"w")
		
		for i in range(self.M):
			for j in range(self.K):
				fout.write("%f " % self.theta[i][j])
			fout.write("\n")
		
		return True

	def save_model_phi(self,filename):
		fout = open(filename,"w")
		
		for i in range(self.K):
			for j in range(self.V):
				fout.write("%f " % self.phi[i][j])
			fout.write("\n")
		
		return True


	def save_model_others(self,filename):
		fout = open(filename,"w")

		fout.write("alpha=%f\n" % self.alpha)
		fout.write("beta=%f\n" % self.beta)
		fout.write("ntopics=%d\n" % self.K)
		fout.write("ndocs=%d\n" % self.M)
		fout.write("nwords=%d\n" % self.V)
		fout.write("liter=%d\n" % self.liter)

		fout.close()

		return True

	def save_model_twords(self,filename):
		fout = open(filename,"w")
		if self.twords > self.V:
			self.twords = self.V
    	
		for k in range(self.K):
			words_probs = []
			for w in range(self.V):
				word_prob = (w,self.phi[k][w])
				words_probs.append(word_prob)

			words_probs = sorted(words_probs,key = lambda x:(float(x[1])),reverse = True)
			

			fout.write("Topic %dth:\n" % k)

			for i in range(self.twords):
				if self.id2word.get(words_probs[i][0]):
					fout.write("\t%s   %f\n" % (self.id2word[words_probs[i][0]],words_probs[i][1]))
	
		fout.close()

		return True

	def save_inf_model(self,model_name):
		if not self.save_inf_model_tassign(self.directory + model_name + self.tassign_suffix):
			return False

		if not self.save_inf_model_others(self.directory + model_name + self.others_suffix):
			return False

		if not self.save_inf_model_newtheta(self.directory + model_name + self.theta_suffix):
			return False

		if not self.save_inf_model_newphi(self.directory + model_name + self.phi_suffix):
			return False

		if not self.twords > 0:
			if self.save_inf_model_twords(self.directory + model_name + self.twords_suffix):
				return False

		return True

	def save_inf_model_tassign(self,filename):
		fout = open(filename,"w")

		for i in range(self.pnewdata.M):
			for j in range(self.pnewdata.docs[i].length):
				fout.write("%d:%d " % (self.pnewdata.docs[i].words[j],self.newz[i][j]))
			fout.write("\n")

		fout.close()
		return True

	def save_inf_model_newtheta(self,filename):
		fout = open(filename,"w")
		
		for i in range(self.newM):
			for j in range(self.K):
				fout.write("%f " % self.newtheta[i][j])
			fout.write("\n")
		
		return True

	def save_inf_model_newphi(self,filename):
		fout = open(filename,"w")
		
		for i in range(self.K):
			for j in range(self.newV):
				fout.write("%f " % self.newphi[i][j])
			fout.write("\n")
		
		return True

	def save_inf_model_others(self,filename):
		fout = open(filename,"w")

		fout.write("alpha=%f\n" % self.alpha)
		fout.write("beta=%f\n" % self.beta)
		fout.write("ntopics=%d\n" % self.K)
		fout.write("ndocs=%d\n" % self.newM)
		fout.write("nwords=%d\n" % self.newV)
		fout.write("liter=%d\n" % self.inf_liter)

		fout.close()

		return True

	def save_inf_model_twords(self,filename):
		fout = open(filename,"w")
		if self.twords > self.newV:
			self.twords = self.newV
		
		for k in range(self.K):
			words_probs = []
			for w in range(self.newV):
				word_prob = (w,self.newphi[k][w])
				words_probs.append(word_prob)

			words_probs = sorted(words_probs,key = lambda x:(float(x[1])),reverse = True)

			fout.write("Topic %dth:\n" % k)
			
			for i in range(self.twords):
				if self.pnewdata._id2id.get(words_probs[i][0]):
					_it = self.pnewdata._id2id[words_probs[i][0]]
					if self.id2word.get(_it):
						fout.write("\t%s   %f\n" % (self.id2word[_it],words_probs[i][1]))
        
		fout.close()
 
		return True

	def init_est(self):
		self.p = np.zeros(self.K)
		self.ptrndata = dataset()
		if not self.ptrndata.read_trndata(self.directory + self.dfile,self.wordmapfile):
			print "Fail to read training data!"
			return False

		self.M = self.ptrndata.M
		self.V = self.ptrndata.V
		
		self.nw = np.zeros((self.V,self.K),dtype=int)
		self.nd = np.zeros((self.M,self.K),dtype=int)

		self.nwsum = np.zeros(self.K,dtype=int)
		self.ndsum = np.zeros(self.M,dtype=int)

		self.z = [0] * self.M

		for m in range(self.ptrndata.M):
			N = self.ptrndata.docs[m].length
			self.z[m] = np.zeros(N,dtype=int)

			for n in range(N):
				topic = int(random.random() * self.K)
				self.z[m][n] = topic

				# word i 被赋予topic j的次数	
				self.nw[self.ptrndata.docs[m].words[n]][topic] += 1
				# document i 中的单词被赋予topic j的次数
				self.nd[m][topic] += 1
				# 被赋予topic j的单词总数
				self.nwsum[topic] += 1

			# document i的单词总数
			self.ndsum[m] = N

		self.theta = np.zeros((self.M,self.K))
		self.phi = np.zeros((self.K,self.V))
	
		return True

	def estimate(self):
		if self.twords > 0:
			self.id2word = dataset.read_id2word(self.directory + self.wordmapfile)


		print "Sampling %d iterations!" % self.niters
		start = time.time()
		last_iter = self.liter

		for self.liter in range(last_iter + 1,self.niters + last_iter + 1):
			print "Iteration %d ... " % self.liter
			for m in range(self.M):
				for n in range(self.ptrndata.docs[m].length):
					topic = self.sampling(m,n)
					self.z[m][n] = topic

			if self.savestep > 0:
				if self.liter % self.savestep == 0:
					print "Saving the model at iteration %d ..." % self.liter
					self.compute_theta()
					self.compute_phi()
					self.save_model(utils.generate_model_name(self.liter))
			duration = time.time() - start
			print "%lf per iteration" % (duration / self.liter)
			
			self.compute_theta()
			self.compute_phi()
			print "Perplexity = %f" % self.calc_est_perplexity()

 		print "Gibbs sampling completed!"
 		print "Saving the final model!"

		self.compute_theta()
		self.compute_phi()
		self.liter -= 1
		self.save_model(utils.generate_model_name(-1))

		print "Perplexity = %f" % self.calc_est_perplexity()
	
	''' Gibbs sampling for sampling topic '''
	def sampling(self,m,n):
		topic = self.z[m][n]
		w = self.ptrndata.docs[m].words[n]
		self.nw[w][topic] -= 1
		self.nd[m][topic] -= 1
		self.nwsum[topic] -= 1
		self.ndsum[m] -= 1

		Vbeta = self.V * self.beta
		Kalpha = self.K * self.alpha
		
		# for k in range(self.K):
		# 	self.p[k] = (self.nw[w][k] + self.beta) / (self.nwsum[k] + Vbeta) * (self.nd[m][k] + self.alpha) / (self.ndsum[m] + Kalpha)

		self.p = (self.nw[w] + self.beta) * (self.nd[m] + self.alpha) / ((self.nwsum + Vbeta) * (self.ndsum[m] + Kalpha))

		for k in range(1,self.K):
			self.p[k] += self.p[k - 1]

		u = random.random() * self.p[self.K - 1]

		for topic in range(self.K):
			if self.p[topic] > u:
				break

		self.nw[w][topic] += 1
		self.nd[m][topic] += 1
		self.nwsum[topic] += 1
		self.ndsum[m] += 1

		return topic

	def compute_theta(self):
		for m in range(self.M):
			for k in range(self.K):
				self.theta[m][k] = (self.nd[m][k] + self.alpha) / (self.ndsum[m] + self.K * self.alpha)


	def compute_phi(self):
		for k in range(self.K):
			for w in range(self.V):
				self.phi[k][w] = (self.nw[w][k] + self.beta) / (self.nwsum[k] + self.V * self.beta)

	def init_inf(self):
		self.p = np.zeros(self.K)

		if not self.load_model(self.model_name):
			print "Fail to load word-topic assignmetn file of the model!"
			return False
	
		self.nw = np.zeros((self.V,self.K),dtype=int)
		self.nd = np.zeros((self.M,self.K),dtype=int)

		self.nwsum = np.zeros(self.K,dtype=int)
		self.ndsum = np.zeros(self.M,dtype=int)

		for m in range(self.ptrndata.M):
			N = self.ptrndata.docs[m].length

			for n in range(N):
				w = self.ptrndata.docs[m].words[n]
				topic = self.z[m][n]

				self.nw[w][topic] += 1
				self.nd[m][topic] += 1
				self.nwsum[topic] += 1

			self.ndsum[m] = N

		self.theta = np.zeros((self.M,self.K))
		self.phi = np.zeros((self.K,self.V))
		self.compute_theta()
		self.compute_phi()
		
		print "model Perplexity = %f" % self.calc_est_perplexity()

		# 读入新数据进行推测
		self.pnewdata = dataset()
		if self.withrawstrs:
			if not self.pnewdata.read_newdata_withrawstrs(self.directory + self.dfile, self.directory + self.wordmapfile):
				print "Fail to read new data!"
				return False
		else: 
			if not self.pnewdata.read_newdata(self.directory + self.dfile,self.directory + self.wordmapfile):
				print "Fail to read new data!"
				return False

		self.newM = self.pnewdata.M
		self.newV = self.pnewdata.V

		self.newnw = np.zeros((self.newV,self.K))
		self.newnd = np.zeros((self.newM,self.K))

		self.newnwsum = np.zeros(self.K)
		self.newndsum = np.zeros(self.newM)
	
		self.newz = [0] * self.newM
		for m in range(self.pnewdata.M):
			N = self.pnewdata.docs[m].length
			self.newz[m] = np.zeros(N)
			for n in range(N):
				w = self.pnewdata.docs[m].words[n]
				_w = self.pnewdata._docs[m].words[n]
				topic = int(random.random() * self.K)
				self.newz[m][n] = topic

				self.newnw[_w][topic] += 1
				self.newnd[m][topic] += 1
				self.newnwsum[topic] += 1

			self.newndsum[m] = N

		self.newtheta = np.zeros((self.newM,self.K))
		self.newphi = np.zeros((self.K,self.newV))

		return True

	def inference(self):
		if self.twords > 0:
			self.id2word = dataset.read_id2word(self.directory + self.wordmapfile)

		print "Sampling %d iterations for inference!" % self.niters

		for self.inf_liter in range(1,self.niters + 1):
			print "Iteration %d ..." % self.inf_liter

			for m in range(self.newM):
				for n in range(self.pnewdata.docs[m].length):
					topic = self.inf_sampling(m,n)
					self.newz[m][n] = topic

		print "Gibbs sampling for inference completed!"
		print "Saving the inference outputs!"
		self.compute_newtheta()
		self.compute_newphi()
		self.inf_liter -= 1
		self.save_inf_model(self.dfile)
		print "Perplexity = %f" % self.calc_inf_perplexity()

	def inf_sampling(self,m,n):
		topic = self.newz[m][n]
		w = self.pnewdata.docs[m].words[n]
		_w = self.pnewdata._docs[m].words[n]
		self.newnw[_w][topic] -= 1
		self.newnd[m][topic] -= 1
		self.newnwsum[topic] -= 1
		self.newndsum[m] -= 1

		Vbeta = self.V * self.beta
		Kalpha = self.K * self.alpha

		for k in range(self.K):
			self.p[k] = (self.nw[w][k] + self.newnw[_w][k] + self.beta) / (self.nwsum[k] + self.newnwsum[k] + Vbeta) * (self.newnd[m][k] + self.alpha) / (self.newndsum[m] + Kalpha)

		for k in range(1,self.K):
			self.p[k] += self.p[k - 1]

		u = random.random() * self.p[self.K - 1]

		for topic in range(self.K):
			if self.p[topic] > u:
				break
	
		self.newnw[_w][topic] += 1;
		self.newnd[m][topic] += 1;
		self.newnwsum[topic] += 1;
		self.newndsum[m] += 1;  
    	
		return topic

	def compute_newtheta(self):
		for m in range(self.newM):
			for k in range(self.K):
				self.newtheta[m][k] = (self.newnd[m][k] + self.alpha) / (self.newndsum[m] + self.K * self.alpha)

	def compute_newphi(self):
		for k in range(self.K):
			for w in range(self.newV):
				if self.pnewdata._id2id.get(w,-1) >= 0:
					self.newphi[k][w] = (self.nw[self.pnewdata._id2id[w]][k] + self.newnw[w][k] + self.beta) / (self.nwsum[k] + self.newnwsum[k] + self.V * self.beta)

	def calc_inf_perplexity(self):
		perplexity = 0.0
		for m in range(self.newM):
		    for w in self.pnewdata._docs[m].words:
		        perplexity += np.log(np.sum(self.newtheta[m] * self.newphi[:,w]))
		return np.exp(-(perplexity / np.sum(self.newndsum)))

	def calc_est_perplexity(self):
		perplexity = 0.0
		for m in range(self.M):
		    for w in self.ptrndata.docs[m].words:
		        perplexity += np.log(np.sum(self.theta[m] * self.phi[:,w]))
		return np.exp(-(perplexity / np.sum(self.ndsum)))