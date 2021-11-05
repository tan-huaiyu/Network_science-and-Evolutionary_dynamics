# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 22:32:50 2020

@author: Huaiyu TAN
"""

import numpy as np

def slnet(x = 4):
    karray,a,b,c,d = [],[],[],[],[]
    x = eval(input("please input a number of Order: \n \n"))
    nebmatrix = np.zeros(shape = (x**2,x**2))
    for i in range(1,x**2 + 1):
        karray.append(i)
    print("\n \n",karray,"\n \n",end = ",")
    i = 1
    while i <= x**2:
        if i-x > 0: a = i-x
        else: a = i+x*x-x
        if i+x < x*x+1: b = i+x
        else: b = i-x*x+x
        if (i-1) % x != 0: c = i-1
        else: c = i+x-1
        if i % x != 0: d = i+1
        else: d = i-x+1       
        print("Number {} ==> ".format(i), "{}, {}, {}, {}".format(a,b,c,d))
        nebarray = np.array([a,b,c,d])
        
        for j in range(1,x**2 + 1):
            if j in nebarray:
                nebmatrix[i-1,j-1] = 1
            else:
                nebmatrix[i-1,j-1] = 0
        i += 1
        continue
    # np.savetxt('Adjacent_Matrix.csv',nebmatrix,delimiter = ',')
    print("\n \n Adjacent Matrix :","\n \n",nebmatrix)
    return nebmatrix

nebmatrix = slnet()