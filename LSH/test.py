



import numpy as np
from sklearn.neighbors import NearestNeighbors
samples = [[0, 0, 2], [1, 0, 0], [0, 0, 1]]
neigh = NearestNeighbors(2)


neigh.fit(samples)
print(neigh.kneighbors([1,1,1], return_distance=True))












#import numpy as np
#from PIL import Image

# Creates a random image 100*100 pixels
#mat=np.zeros((20,20))
#for i in range(0,18):
#    for j in range(0,18):
#        mat[i][j]=255

# Creates PIL image
#img = Image.fromarray(mat,'l')
#img.show()





#from PIL import Image
#import numpy as np
#w, h = 512, 512
#data = np.zeros((h, w, 3), dtype=np.uint8)
#data[256, 256] = [255, 255,256]
#img = Image.fromarray(data, 'RGB')
#img.save('my.png')
#img.show()



#from scipy.spatial import distance
#a = (1,2,3)
#b = (4,5,6)
#dst = distance.euclidean(a,b)
#print(dst)


#x=0
#for i in range(0,400):
#    x=x+(100*100*4)



#print(x)



