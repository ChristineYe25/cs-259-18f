import random
import os

file = open("uf250-test3.cnf","w")
num_var=7
num_clau=100
map = {}
i=0
while(i<100):
    a=random.randint(-7,7)
    b=random.randint(-7,7)
    c=random.randint(-7,7)
    if (abs(a)!=abs(b) & abs(a)!=abs(c) & abs(b)!=abs(c)):
        print(i)
        key=str(a)+str(b)+str(c);
        print(key)
        if key in map:
            continue
        file.write(str(a)+" "+str(b)+" "+ str(c)+"\n")
        map[key]=1
        i+=1
    else :
        continue







