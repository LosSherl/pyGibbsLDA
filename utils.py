#!/usr/bin/python2.7  
# -*- coding: utf-8 -*-

from strtokenizer import strtokenizer

MODEL_STATUS_UNKNOWN = 0
MODEL_STATUS_EST = 1
MODEL_STATUS_INF = 2

def parse_args(argc,argv,pmodel):
	model_status = MODEL_STATUS_UNKNOWN
	directory = ""
	model_name = ""
	dfile = ""
	alpha = -1.0
	beta = -1.0
	K = 0
	niters = 0
	savestep = 0
	twords = 0
	withrawdata = 0

	i = 0
	while i < argc:
		arg = argv[i]
	
		if arg == "-est":
		    model_status = MODEL_STATUS_EST
		elif arg == "-inf":
		    model_status = MODEL_STATUS_INF
		elif arg == "-dir":
		    directory = argv[i + 1]
		    i = i + 1	
		elif arg == "-dfile":
		    dfile = argv[i + 1]
		    i = i + 1	    
		elif arg == "-model":
		    model_name = argv[i + 1]
		    i = i + 1	    	    
		elif arg == "-alpha":
		    alpha = int(argv[i + 1])
		    i = i + 1 
		elif arg == "-beta":
		    beta = int(argv[i + 1])
		    i = i + 1	    
		elif arg == "-ntopics":
		    K = int(argv[i + 1])
		    i = i + 1	    
		elif arg == "-niters":
		    niters = int(argv[i + 1])   
		    i = i + 1
		elif arg == "-savestep":
		    savestep = int(argv[i + 1])
		    i = i + 1
		elif arg == "-twords":
		    twords = int(argv[i + 1])
		    i = i + 1
		elif arg == "-withrawdata":
		    withrawdata = True
		else:
			pass
		i = i + 1

	if model_status == MODEL_STATUS_EST:
		if dfile == "":
		    print "Please specify the input data file for model estimation!\n"
		    return False
		
		pmodel.model_status = model_status
		
		if K > 0:
		    pmodel.K = K
		
		if alpha >= 0.0:
		    pmodel.alpha = alpha
		else:
		    pmodel.alpha = 50.0 / pmodel.K
		
		if beta >= 0.0:
		    pmodel.beta = beta
		
		if niters > 0:
		    pmodel.niters = niters
		
		if savestep > 0:
		    pmodel.savestep = savestep
		
		if twords > 0:
		    pmodel.twords = twords
		
		pmodel.dfile = dfile
		idx = dfile.rfind("/")
		if idx == -1:
		    pmodel.directory = "./"
		else:
		    pmodel.directory = dfile[:idx + 1]
		    pmodel.dfile = dfile[idx + 1:]
		    print "dir = %s\n" % pmodel.directory
		    print "dfile = %s\n" % pmodel.dfile
    
	if model_status == MODEL_STATUS_INF:
		if directory == "":
		    print "Please specify model directory please!\n"
		    return False
		
		if model_name == "":
		    print "Please specify model name for inference!\n"
		    return False

		if dfile == "":
		    print "Please specify the new data file for inference!\n"
		    return False
		
		pmodel.model_status = model_status
		if directory[-1] != '/':
		    directory += "/"

		pmodel.directory = directory
		
		pmodel.model_name = model_name

		pmodel.dfile = dfile

		if niters > 0:
		    pmodel.niters = niters
		else:
		    pmodel.niters = 20
		
		if twords > 0:
		    pmodel.twords = twords
		
		if withrawdata > 0:
		    pmodel.withrawstrs = withrawdata
			
		pmodel = read_and_parse(pmodel.directory + pmodel.model_name + pmodel.others_suffix, pmodel)
		if not pmodel:
		    return False
		
	if model_status == MODEL_STATUS_UNKNOWN:
		print "Please specify the task you would like to perform (-est/-inf)!\n"
		return False
	return pmodel

def read_and_parse(file_name,pmodel):
	fr = open(file_name,"r")

	for line in fr:
		strtok = strtokenizer(line,"=")
		count = strtok.count_tokens()

		if count != 2:
			continue

		optstr = strtok.token(0)
		optval = strtok.token(1)

		if optstr == "alpha":
			pmodel.alpha = float(optval)
		elif optstr == "beta":
			pmodel.beta = float(optval)
		elif optstr == "ntopics":
			pmodel.K = int(optval)
		elif optstr == "ndocs":
			pmodel.M = int(optval)
		elif optstr == "nwords":
			pmodel.V = int(optval)
		elif optstr == "liter":
			pmodel.liter = int(optval)
		else:
			pass

	return pmodel	

def generate_model_name(iter):
	model_name = "model-"

	if 0 <= iter and iter < 0:
		suffix = "0000%d" % iter
	elif 10 <= iter and iter < 100:
		suffix = "000%d" % iter
	elif 100 <= iter and iter < 1000:
		suffix = "00%d" % iter
	elif 1000 <= iter and iter < 10000:
		suffix = "0%d" % iter
	else:
		suffix = "%d" % iter

	if iter >= 0:
		model_name = model_name + suffix
	else:
		model_name = model_name + "final"

	return model_name
