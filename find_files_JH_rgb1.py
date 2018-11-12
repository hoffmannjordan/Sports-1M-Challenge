import numpy as np
from mpi4py import MPI
import glob
import cv2
import os

class Import(object):
    
    def __init__(self):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
    
    def divide_work(self):
        # Initialize MPI    

        directory = '/Users/jordanhoffmann/Dropbox/10_Sports/Test8'

        if self.rank == 0:
            # Find files
            file_list = glob.glob(directory+"/*.mp4")
            #print file_list
            #exit()
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
            x = np.zeros((0,100*100+3*100),dtype=np.float32)
            sports = np.zeros(0,dtype=np.int32)
            for t in tmp:
                x = np.vstack((x,t[0]))
                sports = np.append(sports,t[1])
            np.save('x_rf_rgb10_test_8.npy',x)
            np.save('y_rf_rgb10_test_8.npy',sports)          
  #          np.save('x_train_19_bw100.npy',x)
  #          np.save('y_train_19_bw100.npy',sports)

def extract_image(adress,frames_per_video = 8,frame_n = 50,dim =  100):
    #print 'IN HERE'
    cap = cv2.VideoCapture(adress)
    ALL = np.zeros((frames_per_video,dim**2+3*dim))
    counter_1 = 0
    counter_2 = 0
    loc = 0
    while(True):
        ret, frame = cap.read()
        counter_1 +=1
        #print 'HERE2'
        if counter_1%frame_n == 40:
            frame = cv2.resize(frame,(dim,dim))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = gray.reshape((1,dim**2))
            b,g,r = cv2.split(frame)
            if np.var(gray) > 1500:
	            bm = b.mean(axis=0)
	            gm = g.mean(axis=0)
	            rm = r.mean(axis=0)
	            final_sol = np.concatenate((gray[0],bm,gm,rm))
	            #MAT[counter_2] = gray
	            ALL[loc] = final_sol
	            #print ALL[loc]
	            ALL[loc].shape
	            #exit()
	            counter_2 +=1
	            loc +=1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if counter_2 == frames_per_video:
            break
    return ALL

def work(file_list):
    sports = np.zeros(0,dtype=np.int32)
    x = np.zeros((0,100*100+3*100))
    
    for movie in file_list:
        #print movie.split('_')
        tmp         = movie.split('_')[2]
        #print tmp
        input_movie = extract_image(movie)
        #print input_movie
        #exit()


        if len(tmp.split(','))>1:
            for entry in tmp.split(','):
                pic_per_movie = np.ones(input_movie.shape[0])*int(entry)
                sports = np.append(sports,pic_per_movie)
                x = np.vstack((x,input_movie))
        else:
            pic_per_movie = np.ones(input_movie.shape[0])*int(tmp)
            sports = np.append(sports,pic_per_movie)
            x = np.vstack((x,input_movie))
    
    return x,sports
    
    
imp = Import()
file_list = imp.divide_work()
x,sports = work(file_list) 
#print x.shape
#print sports.shape
imp.reduce_work(x,sports)

