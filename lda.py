#!/usr/bin/python2.7  
# -*- coding: utf-8 -*-

from model import LDA
import sys

MODEL_STATUS_UNKNOWN = 0
MODEL_STATUS_EST = 1
MODEL_STATUS_INF = 2

def show_help():
	print "Command line usage:\n"
	print "\tlda -est -alpha <double> -beta <double> -ntopics <int> -niters <int> -savestep <int> -twords <int> -dfile <string>\n"
	print "\tlda -inf -dir <string> -model <string> -niters <int> -twords <int> -dfile <string>\n"

def main(argc,argv):
	lda = LDA()

	if not lda.init(argc,argv):
		show_help()
		return

	if lda.model_status == MODEL_STATUS_EST:
		lda.estimate()

	if lda.model_status == MODEL_STATUS_INF:
		lda.inference()


if __name__ == '__main__':
	args = sys.argv
	main(len(args),args)
