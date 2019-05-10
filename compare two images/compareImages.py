import cv2
import numpy as np
import scipy
from scipy.misc import imread
import _pickle as pickle
import random
import os
import matplotlib.pyplot as plt
import pdb

class compareTwoImages:
    def __init__(self, imgPath):
        self.basePath = imgPath
        self.image1 = imgPath + 'image1.jpg'
        self.image2 = imgPath + 'image2.jpg'
        
        
    def getImage(self, pickled_db_path="imgFeature.pck"):
        
        img_files = [os.path.join(self.basePath, p) for p in sorted(os.listdir(self.basePath))]
        # getting 3 random images 
        sampleImg = random.sample(img_files, 2)
        
        print(self.image1)
        print(self.image2)
        imgFiles = [self.image1, self.image2]
        imgDict = {}
        for f in imgFiles:
            print('Extracting features from image %s' % f)
            name = f.split('/')[-1].lower()
            print(name)
            imgDict[name] = self.extractImg(f)  
            
        print('final result is:', imgDict[name])
        with open(pickled_db_path, 'wb') as fp:
            pickle.dump(imgDict, fp)
            
        matchImg = self.matchImage('imgFeature.pck') 
        for i in sampleImg:
            print('Query image')
            # self.show_img(i)
            names, match = self.match(i, topn=2)
            print('Result images')
            for j in range(2):
                print('Match %s' % (1-match[j]))
                # self.show_img(os.path.join(self.basePath, names[j]))
    
    
    def show_img(self, path):
        img = imread(path, mode="RGB")
        plt.imshow(img)
        plt.show()    
        
    def extractImg(self, name, vector_size=32): 
        image = imread(name, mode="RGB")
        try :
            alg = cv2.KAZE_create()
            kps = alg.detect(image)
            kps = sorted(kps, key=lambda x: -x.response)[:vector_size]
            kps, dsc = alg.compute(image, kps)

            dsc = dsc.flatten()
            needed_size = (vector_size * 64)
            if dsc.size < needed_size:
                dsc = np.concatenate([dsc, np.zeros(needed_size - dsc.size)])

        except cv2.error as e:
            print('Error: ', e)
            return None

        return dsc

    def matchImage(self, img_loc):
        with open(img_loc, 'rb') as fp:
            self.data = pickle.load(fp)
        self.names = []
        self.matrix = []
        for k, v in self.data.items():
            self.names.append(k)
            self.matrix.append(v)
        self.matrix = np.array(self.matrix)
        self.names = np.array(self.names)
    
        
    def cos_cdist(self, vector):
        # getting cosine distance between search image and images database
        v = vector.reshape(1, -1)
        return scipy.spatial.distance.cdist(self.matrix, v, 'cosine').reshape(-1)

    def match(self, image_path, topn=5):
        features = self.extractImg(image_path)
        img_distances = self.cos_cdist(features)
        # getting top 5 records
        nearest_ids = np.argsort(img_distances)[:topn].tolist()
        nearest_img_paths = self.names[nearest_ids].tolist()

        return nearest_img_paths, img_distances[nearest_ids].tolist()
        
        
        
        
CI = compareTwoImages(r'./img/')
CI.getImage()














