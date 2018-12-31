import pandas as pd
import numpy as np
import csv
from scipy.spatial import distance
from PIL import Image






csv = np.genfromtxt ('pat.csv', delimiter=",")

a=csv[:,0]

im=np.zeros((20,20))
im1=np.zeros((20,20))



for i in range(0,20):
    for j in range(0,20):
        im[i][j]=csv[j*20+i][2046]



for i in range(0,20):
    for j in range(0,20):
        im1[i][j]=csv[j*20+i][17951]




img = Image.fromarray(im)
img.show()


img1 = Image.fromarray(im1)
img1.show()



print(type(csv))

print(csv.shape)




#data=pd.read_csv("pat.csv")




#print(data.iat(0,0))

