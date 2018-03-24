#!/usr/bin/python2.7  
# -*- coding: utf-8 -*-

import random
import utils
from strtokenizer import strtokenizer
from dataset import dataset
from dataset import document
from datetime import datetime

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

	def init(self,argc,argv):
		if self.parse_args(argc,argv):
			return 1

		if self.model_status == MODEL_STATUS_EST:
			if self.init_est(): 
				return 1
		elif self.model_status == MODEL_STATUS_ESTC:
			if self.init_estc():
				return 1
		elif self.model_status == MODEL_STATUS_INF:
			if self.init_inf():
				return 1

		return 0

	def load_model(self,model_name):

		filename = self.directory + model_name + self.tassign_suffix;
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
				fout.write("%f " % self.theta[i][j])
			fout.write("\n")
		
		return 0

	def save_model_phi(self,filename):
		fout = open(filename,"w")
		
		for i in range(self.K):
			for j in range(self.V):
				fout.write("%f " % self.phi[i][j])
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

			words_probs = sorted(words_probs,key = lambda x:(float(x[1])),reverse = True)
			# utils.quicksort(w0ords_probs,0,len(words_probs) - 1)

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
				fout.write("%f " % self.newtheta[i][j])
			fout.write("\n")
		
		return 0

	def save_inf_model_newphi(self,filename):
		fout = open(filename,"w")
		
		for i in range(self.K):
			for j in range(self.newV):
				fout.write("%f " % self.newphi[i][j])
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

			# utils.quicksort(words_probs,0,len(words_probs) - 1)
			words_probs = sorted(words_probs,key = lambda x:(float(x[1])),reverse = True)

			fout.write("Topic %dth:\n" % k)
			
			for i in range(self.twords):
				if self.pnewdata._id2id.get(words_probs[i][0]):
					_it = self.pnewdata._id2id[words_probs[i][0]]
					if self.id2word.get(_it):
						fout.write("\t%s   %f\n" % (self.id2word[_it],words_probs[i][1]))
        
		fout.close()
 
		return 0

	def init_est(self):
		self.p = [0] * self.K

		self.ptrndata = dataset()
		if self.ptrndata.read_trndata(self.directory + self.dfile,self.wordmapfile):
			print "Fail to read training data!"
			return 1

		self.M = self.ptrndata.M
		self.V = self.ptrndata.V
		
		self.nw = [0] * self.V

		for w in range(self.V):
			self.nw[w] = [0] * self.K
			for k in range(self.K):
				self.nw[w][k] = 0

		self.nd = [0] * self.M

		for m in range(self.M):
			self.nd[m] = [0] * self.K
			for k in range(self.K):
				self.nd[m][k] = 0
	
		self.nwsum = [0] * self.K

		for k in range(self.K):
			self.nwsum[k] = 0
	
		self.ndsum = [0] * self.M

		for m in range(self.M):
			self.ndsum[m] = 0

		self.z = [0] * self.M

		for m in range(self.ptrndata.M):
			N = self.ptrndata.docs[m].length
			self.z[m] = [0] * N

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
	
		self.theta = [0] * self.M
		for m in range(self.M):
			self.theta[m] = [0] * self.K 
		
		self.phi = [0] * self.K
		for k in range(self.K):
			self.phi[k] = [0] * self.V
	
	
		return 0

  	def init_estc(self):
  		self.p = [0] * self.K

  		if self.load_model(self.model_name):
  			print "Fail to load word-topic assignmetn file of the model!\n"
  			return 1

  		self.nw = [0] * self.V
  		for w in range(self.V):
  			self.nw[w] = [0] * self.K
  			for k in range(self.K):
  				self.nw[w][k] = 0

  		self.nd = [0] * self.M
  		for m in range(self.M):
  			self.nd = [0] * self.K
			for k in range(self.K):
				self.nd[m][k] = 0

		self.nwsum = [0] * self.K
		for k in range(self.K):
			self.nwsum[k] = 0
		
		self.ndsum = [0] * self.M
		for m in range(self.M):
			self.ndsum[m] = 0
    	
		for m in range(self.ptrndata.M):
			N = self.ptrndata.docs[m].length
			for n in range(N):
				w = self.ptrndata.docs[m].words[n]
				topic = self.z[m][n]

				# word i被赋予topic j的次数
				self.nw[w][topic] += 1
				# document i中的单词被赋予topic j的次数
				self.nd[m][topic] += 1
				# 被赋予topic j的单词总数
				self.nwsum[topic] += 1

    		# document i的单词总数
    		self.ndsum[m] = N

		self.theta = [0] * self.M
		for m in range(self.M):
			self.theta[m] = [0] * self.K

		self.phi = [0] * self.K
		for k in range(self.K):
			self.phi[k] = [0] * self.V
	
		return 0

	def estimate(self):
		if self.twords > 0:
			self.id2word = dataset.read_id2word(self.directory + self.wordmapfile)


		print "Sampling %d iterations!" % self.niters

		last_iter = self.liter

		for self.liter in range(last_iter + 1,self.niters + last_iter + 1):
			print "Iteration %d ..." % self.liter
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


 		print "Gibbs sampling completed!"
 		print "Saving the final model!"

		self.compute_theta()
		self.compute_phi()
		self.liter -= 1
		self.save_model(utils.generate_model_name(-1))

	def sampling(self,m,n):
		topic = self.z[m][n]
		w = self.ptrndata.docs[m].words[n]
		self.nw[w][topic] -= 1
		self.nd[m][topic] -= 1
		self.nwsum[topic] -= 1
		self.ndsum[m] -= 1

		Vbeta = self.V * self.beta
		Kalpha = self.K * self.alpha
		
		for k in range(self.K):
			self.p[k] = (self.nw[w][k] + self.beta) / (self.nwsum[k] + Vbeta) * (self.nd[m][k] + self.alpha) / (self.ndsum[m] + Kalpha)

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
		self.p = [0] * self.K

		if self.load_model(self.model_name):
			print "Fail to load word-topic assignmetn file of the model!"
			return 1
	
		self.nw = [0] * self.V
		for w in range(self.V):
			self.nw[w] = [0] * self.K
			for k in range(self.K):
				self.nw[w][k] = 0

		self.nd = [0] * self.M
		for m in range(self.M):
			self.nd[m] = [0] * self.K
			for k in range(self.K):
				self.nd[m][k] = 0

		self.nwsum = [0] * self.K
		for k in range(self.K):
			self.nwsum[k] = 0
	
		self.ndsum = [0] * self.M
		for m in range(self.M):
			self.ndsum[m] = 0

		for m in range(self.ptrndata.M):
			N = self.ptrndata.docs[m].length

			for n in range(N):
				w = self.ptrndata.docs[m].words[n]
				topic = self.z[m][n]

				self.nw[w][topic] += 1
				self.nd[m][topic] += 1
				self.nwsum[topic] += 1

			self.ndsum[m] = N

		# 读入新数据进行推测
		self.pnewdata = dataset()
		if self.withrawstrs:
			if self.pnewdata.read_newdata_withrawstrs(self.directory + self.dfile, self.directory + self.wordmapfile):
				print "Fail to read new data!"
		else: 
			if self.pnewdata.read_newdata(self.directory + self.dfile,self.directory + self.wordmapfile):
				print "Fail to read new data!"
				return 1

		self.newM = self.pnewdata.M
		self.newV = self.pnewdata.V

		self.newnw = [0] * self.newV
		for w in range(self.newV):
			self.newnw[w] = [0] * self.K
			for k in range(self.K):
				self.newnw[w][k] = 0
		
		self.newnd = [0] * self.newM
		for m in range(self.newM):
			self.newnd[m] = [0] * self.K
			for k in range(self.K):
				self.newnd[m][k] = 0

		self.newnwsum = [0] * self.K
		for k in range(self.K):
			self.newnwsum[k] = 0
		
		self.newndsum = [0] * self.newM
		for m in range(self.newM):
			self.newndsum[m] = 0
	
		self.newz = [0] * self.newM
		for m in range(self.pnewdata.M):
			N = self.pnewdata.docs[m].length
			self.newz[m] = [0] * N
			for n in range(N):
				w = self.pnewdata.docs[m].words[n]
				_w = self.pnewdata._docs[m].words[n]
				topic = int(random.random() * self.K)
				self.newz[m][n] = topic

				self.newnw[_w][topic] += 1
				self.newnd[m][topic] += 1
				self.newnwsum[topic] += 1

			self.newndsum[m] = N

		self.newtheta = [0] * self.newM
		for m in range(self.newM):
			self.newtheta[m] = [0] * self.K
	
		self.newphi = [0] * self.K
		for k in range(self.K):
			self.newphi[k] = [0] * self.newV

		return 0;        

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

		u = random.random() * self.p[k - 1]

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
				if self.pnewdata._id2id.get(w):
					self.newphi[k][w] = (self.nw[self.pnewdata._id2id[w]][k] + self.newnw[w][k] + self.beta) / (self.nwsum[k] + self.newnwsum[k] + self.V * self.beta)

