
import pandas as pd
import numpy as np
import csv
from scipy.spatial import distance
from PIL import Image
from sklearn.neighbors import NearestNeighbors



csv = np.genfromtxt ('/Users/amirhossein/Desktop/term 7/Big_Data/projects/HW1/pat.csv', delimiter=",")

csv=np.transpose(csv)


lsh=np.zeros((59500,20))

for i in range(0, 20):
    a = np.random.normal(size=400)
    w = np.linalg.norm(a)

    for j in range(0, 59500):
        x = 0
        q = 0

        for k in range(0, 400):
            x = x + a[k] * csv[j][k]
        q = int((x / w))
        lsh[j][i] = q




print(lsh[59490])




print(lsh_knn[11])
print(normal_knn[11])




from scipy.spatial import distance
print(distance.euclidean(csv[2046],csv[17951]))
print(distance.euclidean(csv[2000],csv[22804]))

from PIL import Image

im = np.zeros((20, 20))
im1 = np.zeros((20, 20))

for i in range(0, 20):
    for j in range(0, 20):
        im[i][j] = csv[10000][i * 20 + j]

for i in range(0, 20):
    for j in range(0, 20):
        im1[i][j] = csv[15952][i * 20 + j]

img = Image.fromarray(im)
img.show()

img1 = Image.fromarray(im1)
img1.show()

a = np.random.normal(size=400)
print(np.linalg.norm(csv[3000]))

print(np.linalg.norm(csv[39412]))

print(np.linalg.norm(csv[5707]))

w = np.linalg.norm(a)

x = 0
y = 0
z = 0

for k in range(0, 400):
    x = x + a[k] * csv[3000][k]
    y = y + a[k] * csv[39412][k]
    z = z + a[k] * csv[5707][k]

print(x / w)

print(y / w)

print(z / w)

print()

print(lsh[25130])
print(lsh[2000])
print(lsh[22804])

lsh_knn = np.zeros((2000, 2000))

neigh = NearestNeighbors(2000)
neigh.fit(lsh)

for i in range(0, 500):
    lsh_knn[i] = ((neigh.kneighbors(lsh[i]))[1])

normal_knn = np.zeros((500, 6))

neigh = NearestNeighbors(6)
neigh.fit(csv)

# neigh.kneighbors(csv[30], return_distance=True)


for i in range(0, 500):
    normal_knn[i] = ((neigh.kneighbors(csv[i]))[1])

# print(normal_knn[1])





print(lsh[1000])
print(lsh[33063])
print(lsh[14737])

y = 0
bond_number = 10
bond_size = 2
bocket_size = 50
number = 0
number_2 = 0
a = []
all

for q in range(1, 100):

    a = []

    for i in range(0, 59500):
        y = 0
        for j in range(0, bond_number):
            for k in range(0, bond_size):
                if (int((lsh[q][j * bond_size + k]) / bocket_size) == int((lsh[i][j * bond_size + k]) / bocket_size)):
                    # if(abs(int((lsh[2000][j*bond_size+k])/bocket_size)-int((lsh[i][j*bond_size+k])/bocket_size)))<=50:
                    y = y + 1
                    break;

        if (y == bond_number):
            a.append(i)

            number = number + 1

    for u in range(1, 11):
        if (normal_knn[q][u] in a):
            number_2 = number_2 + 1

    print(a)

# print(number)
# print(a)
print(number_2)

w = 0
for i in range(0, 500):
    for j in range(1, 6):
        if (normal_knn[i][j] in lsh_knn[i]):
            w = w + 1

print(w)





