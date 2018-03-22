#!/usr/bin/python2.7  
# -*- coding: utf-8 -*-

from model import model
import sys

MODEL_STATUS_UNKNOWN = 0
MODEL_STATUS_EST = 1
MODEL_STATUS_ESTC = 2
MODEL_STATUS_INF = 3

def show_help():
	print "Command line usage:\n"
	print "\tlda -est -alpha <double> -beta <double> -ntopics <int> -niters <int> -savestep <int> -twords <int> -dfile <string>\n"
	print "\tlda -estc -dir <string> -model <string> -niters <int> -savestep <int> -twords <int>\n"
	print "\tlda -inf -dir <string> -model <string> -niters <int> -twords <int> -dfile <string>\n"
	# print "\tlda -inf -dir <string> -model <string> -niters <int> -twords <int> -dfile <string> -withrawdata\n"

def main(argc,argv):
	lda = model()

	if lda.init(argc,argv):
		show_help()
		return 1

	if lda.model_status == MODEL_STATUS_EST or lda.model_status == MODEL_STATUS_ESTC:
		lda.estimate()

	if lda.model_status == MODEL_STATUS_INF:
		lda.inference()


if __name__ == '__main__':
	args = sys.argv
	main(len(args),args)
