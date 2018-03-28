import numpy as np
fin = open("model-final.theta","r")

cnt = 1
for line in fin:
	probs = line.split()
	ps = np.zeros(100)
	for i in range(len(probs)):
		ps[i] = float(probs[i])
	m = np.max(ps)
	print cnt," ",m
	cnt += 1