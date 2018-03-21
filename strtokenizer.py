#!/usr/bin/python2.7  
# -*- coding: utf-8 -*-

class strtokenizer(object):

	def __init__(self,string,seperator = " "):
		self.idx = 0
		parse(string,seperator)
		self.cnt = len(self.tokens)

	def parse(self,string,seperator = " "):
		if seperator != " ":
			self.tokens = string.split(seperator)
		else:
			self.tokens = string.split()

	def count_tokens(self):
		return self.cnt

	def next_token(self):
		if self.idx >= 0 and self.idx < self.cnt:
			self.idx = self.idx + 1
			return self.tokens[self.idx - 1]
		else:
			return ""

	def token(i):
		if i >= 0 and i < self.cnt:
			return self.tokens[i]
		else:
			return " "
		