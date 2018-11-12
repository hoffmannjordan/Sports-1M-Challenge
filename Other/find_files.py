import numpy as np
from mpi4py import MPI
import glob
import cv2
import os

class Import(object):
    
    def __init__(self,dim = 100, colors = 3,frames_per_video = 10):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        
        self.dim = dim 
        self.colors = colors
        self.frames_per_video = frames_per_video
    
    def divide_work(self):
        # Initialize MPI    

        #directory = '/Volumes/Thomasina/CS_205'
        directory = '/Users/hallvardmoiannydal/Dropbox/Shared/Test_Videos'

        if self.rank == 0:
            # Find files
            file_list = glob.glob(directory+"/*.mp4")
            #file_list = file_list[0:4]
    
            # Divide work
            perWorker = int(np.ceil(len(file_list)/float(self.size)))
            datalist = []
            for n in range(self.size):
                if n != (self.size-1):
                    datalist.append(file_list[n*perWorker:(n+1)*perWorker])
                else:
                    datalist.append(file_list[n*perWorker:len(file_list)])
        else:
            datalist = None

        return self.comm.scatter(datalist, root=0)


    def reduce_work(self,x,sports):
        tmp = self.comm.reduce ([[x,sports]],root=0)

        if self.rank == 0:
            for col in xrange(self.colors):
                x_tmp = np.zeros((0,self.dim**2),dtype=np.float32)
            
                if col==0: 
                    sports = np.zeros(0,dtype=np.int32)
                for t in tmp:
                    x_tmp = np.vstack((x_tmp,t[0][col]))
                    
                    if col == 0:
                        sports = np.append(sports,t[1])
                
                if col == 0:
                    x = np.zeros((self.colors,x_tmp.shape[0],x_tmp.shape[1]))
                
                x[col] = x_tmp
                
            x_new = np.zeros((x.shape[1],x.shape[0],x.shape[2]))
            
            for col in xrange(x.shape[0]):
                for row in xrange(x.shape[1]):
                    x_new[row,col] = x[col,row]
            
            np.save('x.npy',x_new)
            np.save('y.npy',sports)

    def extract_image(self,adress,frame_n = 50):
        
    
        cap = cv2.VideoCapture(adress)
        ALL = np.zeros((3,self.frames_per_video,self.dim**2))
        counter_1 = 0
        counter_2 = 0
    
        loc = 0
        while(True):
            ret, frame = cap.read()
            counter_1 +=1
            if counter_1%frame_n == 0:
                frame = cv2.resize(frame,(self.dim,self.dim))
                b,g,r = cv2.split(frame)
                #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                #gray = gray.reshape((1,dim**2))
                #MAT[counter_2] = gray
                ALL[0,loc] = r.reshape((1,self.dim**2))
                ALL[1,loc] = g.reshape((1,self.dim**2))
                ALL[2,loc] = b.reshape((1,self.dim**2))
                counter_2 +=1
                loc +=1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if counter_2 == self.frames_per_video:
                break
        return ALL

    def work(self,file_list):
        sports = np.zeros(0,dtype=np.int32)
    
        for movie in file_list:
            tmp         = movie.split('_')[2]
 
            if len(tmp.split(','))>1:
                for entry in tmp.split(','):
                    pic_per_movie = np.ones(self.frames_per_video)*int(entry)
                    sports = np.append(sports,pic_per_movie)
            else:
                pic_per_movie = np.ones(self.frames_per_video)*int(tmp)
                sports = np.append(sports,pic_per_movie)
            
        x = np.zeros((self.colors,sports.size,self.dim*self.dim))
            
        counter = 0
        for movie in file_list:
            input_movie = self.extract_image(movie)
        
            if len(tmp.split(','))>1:
                for entry in tmp.split(','):
                
                    for col in xrange(self.colors):
                        x[col,counter:counter+self.frames_per_video] = input_movie[col]
                
                    counter += self.frames_per_video
            else:
                for col in xrange(self.colors):
                    x[col,counter:counter+self.frames_per_video] = input_movie[col]
            
                counter += self.frames_per_video
                
    
        return x,sports
    
    
imp = Import()
file_list = imp.divide_work()
x,sports = imp.work(file_list) 
imp.reduce_work(x,sports)

