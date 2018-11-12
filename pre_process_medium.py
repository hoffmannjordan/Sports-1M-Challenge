#import matplotlib
#import matplotlib.pyplot as plt
#matplotlib.pyplot.gray()

import numpy as np
import theano
import theano.tensor as T
from theano import shared, function
theano.config.floatX = 'float32'
rng = np.random.RandomState(42)

# THEANO_FLAGS= 'openmp=True' python ConvolutionalNetworkdatad
import sys
import time

#sys.path.insert(1,'DeepLearningTutorials/code')
#sys.path
import os

class Medium(object):
    
    def __init__(self,load_in=False):
        self.load_in = load_in
        
    def run(self):

        if self.load_in == True:
            number = 300
    
        else:
            import glob
            
            
            def load(directory):
                files = glob.glob(directory+"/*.npy")
                input_number = len(files)/2
            
                letter = []
                index = []
                for file in files:
                    letter.append(file.split('/')[-1].split('.')[0].split('_')[0])
                    index.append(int(file.split('/')[-1].split('.')[0].split('_')[2]))

            
                a = False

                x = np.zeros((0,100*100))
                y = np.zeros(0)
            
                for n in xrange(len(files)):
                    if letter[n] == 'x':
                        m = 0
                        flag = False
                        while flag ==False:
                            if letter[m] == 'y':
                                if (index[n] == index[m]):
                                    flag = True
                                else:
                                    m += 1
                            else:
                                m += 1
                            
                        if a == False:
                            print files[n]
                            temp_x = np.load(files[n])
                            temp_y = np.load(files[m])

                            x = np.vstack((x,temp_x))
                            y = np.append(y,temp_y)

                            del temp_x,temp_y
                        else:
                            temp_x = np.load(files[n])
                            temp_y = np.load(files[m])
                            xt = x 
                            x = np.zeros((x.shape[1]+temp_x.shape[1],x.shape[1]))
                            x[:,:xt.shape[1]] = xt
                            x[:,xt.shape[1]:] = temp_x
                            y = np.append(y,temp_y)
                            del xt,temp_x,temp_y
                
                return x,y
                
            train_set_x,train_set_y = load('../train_data_medium')            
            test_set_x,test_set_y = load('../test_data_medium')  
            
            #train_x = np.load('x_small_test.npy')
            #train_y = np.load('y_small_test.npy')
            #test_x  = np.load('/Users/hallvardmoiannydal/Dropbox/shared/ComputeFest2015_DeepLearning/x_vsmall.npy') 
            #test_y  = np.load('/Users/hallvardmoiannydal/Dropbox/shared/ComputeFest2015_DeepLearning/y_vsmall.npy')
            
            
            #min_y = np.min(y)
            #if min_y>0:
            #    y -= min_y
                
            #lim = 4000
            valid_set_size = 10000
            #flip_prob = 0.5
            
            #print 'Size of training/test-set: ',lim,'/',x.shape[1]-lim
            #rand = np.random.permutation(range(x.shape[1]))
            #a = rand[:lim]
            #b = rand[lim:]
            
            #train_set_x = np.zeros(x[:,:lim].shape)
            #train_set_y = np.zeros(y[:lim].shape)
            #test_set_x = np.zeros(x[:,lim:].shape)
            #test_set_y = np.zeros(y[lim:].shape)
            
            #for n in xrange(len(a)):
            #    train_set_x[:,n,:] = x[:,a[n],:]
            #    train_set_y[n]     = y[a[n]]
            #for n in xrange(len(b)):
            #    test_set_x[:,n,:] = x[:,b[n],:]
            #    test_set_y[n]     = y[b[n]]
            
            #del x,y
            
            rand_val = np.random.permutation(range(test_set_x.shape[0]))[:valid_set_size]
            valid_set_x = np.zeros((valid_set_size,train_set_x.shape[1]))
            valid_set_y = np.zeros(valid_set_size)
            for n in xrange(len(rand_val)):
                valid_set_x[n,:] = test_set_x[rand_val[n],:]
                valid_set_y[n]     = test_set_y[rand_val[n]]
        
        #number_flips = np.int(np.floor(train_set_x.shape[1]*flip_prob))
        #rand = np.random.permutation(range(train_set_x.shape[1]))[:number_flips]
        #
        #for n in xrange(rand.size):
        #    temp = train_set_x[:,rand[n]]
        #    shape = int(np.sqrt(temp.shape[1]))
        #    temp = temp.reshape(3,shape,shape)
        #    temp = temp[:,::-1,:]
        #    temp = temp.reshape(3,shape*shape)
        #    train_set_x[:,rand[n]] = temp
            
    
        train_set_x = train_set_x.astype(np.float32)
        test_set_x = test_set_x.astype(np.float32)
        valid_set_x = valid_set_x.astype(np.float32)

        train_set_y = train_set_y.astype(np.int32)
        test_set_y = test_set_y.astype(np.int32)
        valid_set_y = valid_set_y.astype(np.int32)

        # estimate the mean and std dev from the training data
        # then use these estimates to normalize the data
        # estimate the mean and std dev from the training data
        
        #for n in xrange(train_set_x.shape[0]):
        norm_mean = train_set_x.mean()
        train_set_x = train_set_x - norm_mean
        norm_std = train_set_x.std()
        norm_std = norm_std.clip(0.00001, norm_std)
        train_set_x = train_set_x / norm_std 

        test_set_x = test_set_x - norm_mean
        test_set_x = test_set_x / norm_std 
        valid_set_x = valid_set_x - norm_mean
        valid_set_x = valid_set_x / norm_std 
        
        list_it = [train_set_x,valid_set_x,test_set_x,train_set_y,valid_set_y,test_set_y]
        
        return list_it

if __name__ == "__main__":
    pre_process =  PreProcessSmall()
    pre_process.run()
    
