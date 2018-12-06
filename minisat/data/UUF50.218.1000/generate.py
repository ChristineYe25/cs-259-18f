#!/usr/bin/env python
import os

f = open("run_all", "w")
exe = './minisat'

for file in os.listdir("."):
	if ".cnf" in file: 
		f.write(exe + " " + file+"\n")
