import matplotlib
import matplotlib.pyplot as plt
matplotlib.pyplot.gray()

import numpy as np
import theano
import theano.tensor as T
from theano import shared, function
theano.config.floatX = 'float32'
rng = np.random.RandomState(42)

import sys
import time

#sys.path.insert(1,'DeepLearningTutorials/code')
#sys.path
import os

class PreProcess(object):
    
    def __init__(self,load_in=False):
        self.load_in = load_in
        
    def run(self):

        if self.load_in == True:
            number = 300
            # tmp = np.zeros((5*number,100**2))
            # j=1
            # for i in xrange(1,number+1):
            #     file=np.genfromtxt('../Beach_Soccer/BS_'+str(i)+'.csv',delimiter=',',dtype=None)
            #     file = file.reshape((1,100**2))
            #     tmp[number*(j-1)+i-1] = file
            # j=2
            # for i in xrange(1,number+1):
            #     file=np.genfromtxt('../Bowling/BO_'+str(i)+'.csv',delimiter=',',dtype=None)
            #     file = file.reshape((1,100**2))
            #     tmp[number*(j-1)+i-1] =file
            # j=3
            # for i in xrange(1,number+1):
            #     file=np.genfromtxt('../Karate/KA_'+str(i)+'.csv',delimiter=',',dtype=None)
            #     file = file.reshape((1,100**2))
            #     tmp[number*(j-1)+i-1] = file
            # j=4
            # for i in xrange(1,number+1):
            #     file=np.genfromtxt('../Rugby/RUGBY_'+str(i)+'.csv',delimiter=',',dtype=None)
            #     file = file.reshape((1,100**2))
            #     tmp[number*(j-1)+i-1] = file
            # j=5
            # for i in xrange(1,number+1):
            #     file=np.genfromtxt('../Kayaking/KAYAK_'+str(i)+'.csv',delimiter=',',dtype=None)
            #     file = file.reshape((1,100**2))
            #     tmp[number*(j-1)+i-1] = file
            #
            #
            # tmp_y = np.zeros(tmp.shape[0])
            # for n in xrange(tmp_y.size):
            #     tmp_y[n] = np.floor(n/float(number))
            #
            # np.save('tmp.npy',tmp)
            # np.save('tmp_y.npy',tmp_y)
    
        else:
            import os
            os.chdir('/Users/jordanhoffmann/Dropbox/CS_205_Final_Project')
            tmp = np.load('x_small_test.npy')
            tmp_y = np.load('y_small_test.npy')
            tmp2 = np.load('x_vsmall.npy')
            tmp_y2 = np.load('y_vsmall.npy')
        
        min_v = np.min(tmp_y)
        if min_v>0:
            tmp_y -= min_v
        min_v2 = np.min(tmp_y2)
        if min_v2>0:
            tmp_y2 -= min_v2
        mod_n=3
        
        #train_x= np.zeros((tmp.shape[1],tmp.shape[0],tmp.shape[2]))
        #test_x= np.zeros((tmp.shape[1],tmp.shape[0],tmp.shape[2]))
        #
        #for n in xrange(tmp.shape[0]):
        #    for m in xrange(tmp.shape[1]):
        #        train_x[m,n] = tmp[n,m]
        #        test_x[m,n] = tmp[n,m]   
        
        train_x,test_x = tmp,tmp2           

        #np.zeros((5*number/mod_n,100**2))

        train_y = tmp_y# np.zeros(5*number/mod_n*(mod_n-1))
        test_y = tmp_y2#np.zeros(5*number/mod_n)

        # count_train = 0
        # count_test = 0
        # for n in xrange(tmp_y.size):
        #     if n%mod_n==0:
        #         test_x[count_test] = tmp[n]
        #         test_y[count_test] = tmp_y[n]
        #         count_test += 1
        #
        #     else:
        #         train_x[count_train] = tmp[n]
        #         train_y[count_train] = tmp_y[n]
        #         count_train += 1

        train_set_x, train_set_y = train_x,train_y
        test_set_x, test_set_y = test_x,test_y
        
        #del train_x,train_y
        
        vs = 50 #valid set size
        a = np.random.permutation(range(train_x.shape[0]))[0:vs]
        
        #valid_set_x = np.zeros((vs,train_set_x.shape[1],train_set_x.shape[2]))
        #valid_set_y = np.zeros(vs)
        #for n in xrange(len(a)):
        #    valid_set_x[n][:] = train_set_x[a[n]][:]
        #    valid_set_y[n] += train_set_y[a[n]]
            
        valid_set_x = train_set_x[:,:vs]
        valid_set_y = train_set_y[:vs]
    
        train_set_x = train_set_x.astype(np.float32)
        test_set_x = test_set_x.astype(np.float32)
        valid_set_x = valid_set_x.astype(np.float32)

        train_set_y = train_set_y.astype(np.int32)
        test_set_y = test_set_y.astype(np.int32)
        valid_set_y = valid_set_y.astype(np.int32)

        # estimate the mean and std dev from the training data
        # then use these estimates to normalize the data
        # estimate the mean and std dev from the training data
        
        for n in xrange(train_set_x.shape[0]):
            norm_mean = train_set_x[n].mean()
            train_set_x[n] = train_set_x[n] - norm_mean
            norm_std = train_set_x[n].std()
            norm_std = norm_std.clip(0.00001, norm_std)
            train_set_x[n] = train_set_x[n] / norm_std 

            test_set_x[n] = test_set_x[n] - norm_mean
            test_set_x[n] = test_set_x[n] / norm_std 
            valid_set_x[n] = valid_set_x[n] - norm_mean
            valid_set_x[n] = valid_set_x[n] / norm_std 
        

        train_set_x = theano.shared(train_set_x)
        valid_set_x = theano.shared(valid_set_x)
        test_set_x = theano.shared(test_set_x)

        train_set_y = theano.shared(train_set_y)
        valid_set_y = theano.shared(valid_set_y)
        test_set_y = theano.shared(test_set_y)
        
        list_it = [train_set_x,valid_set_x,test_set_x,train_set_y,valid_set_y,test_set_y]
        
        return list_it
        
        

#from utils import tile_raster_images
#samples = tile_raster_images(train_set_x.get_value(), img_shape=(28,28), tile_shape=(3,10), tile_spacing=(0, 0),
#                       scale_rows_to_unit_interval=True,
#                       output_pixel_vals=True)
#plt.imshow(samples)
#plt.show()

# Setup 1: parameters of the network and training
from convolutional_mlp import LeNetConvPoolLayer
from mlp import HiddenLayer
from logistic_sgd import LogisticRegression

def run():
    
    preProcess = PreProcess()
    data = preProcess.run()
    
    train_set_x,train_set_y = data[0],data[3]
    valid_set_x,valid_set_y = data[1],data[4]#data[1],data[4]
    test_set_x,test_set_y = data[2],data[5]
    
    # network parameters
    num_kernels = [10,10]
    kernel_sizes = [(9, 9), (5, 5)]
    #exit()
    
    # training parameters
    learning_rate = 0.005
    batch_size = 50
    n_sports = np.max(train_set_y.eval())+1
    sigmoidal_output_size = 20
    
    if valid_set_y.eval().size<batch_size:
        print 'Error: Batch size is larger than size of validation set.'

    # Setup 2: compute batch sizes for train/test/validation
    # borrow=True gets us the value of the variable without making a copy.
    n_train_batches = train_set_x.get_value(borrow=True).shape[1]
    n_test_batches = test_set_x.get_value(borrow=True).shape[1]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[1]
    n_train_batches /= batch_size
    n_test_batches /= batch_size
    n_valid_batches /= batch_size


    # Setup 3.
    # Declare inputs to network - x and y are placeholders
    # that will be used in the training/testing/validation functions below.
    x = T.tensor3('x')  # input image data
    y = T.ivector('y') # input label data

    # ## Layer 0 - First convolutional Layer
    # The first layer takes **`(batch_size, 1, 28, 28)`** as input, convolves it with **10** different **9x9** filters, and then downsamples (via maxpooling) in a **2x2** region.  Each filter/maxpool combination produces an output of size **`(28-9+1)/2 = 10`** on a side.
    # The size of the first layer's output is therefore **`(batch_size, 10, 10, 10)`**. 
    
    class Convolution(object):
        
        def __init__(self,batch_size,num_kernels,kernel_sizes,channel):
            
            self.layer0_input_size = (batch_size, 1, 100, 100)  # fixed size from the data
            self.edge0 = (100 - kernel_sizes[0][0] + 1) / 2
            self.layer0_output_size = (batch_size, num_kernels[0], self.edge0, self.edge0)
            # check that we have an even multiple of 2 before pooling
            assert ((100 - kernel_sizes[0][0] + 1) % 2) == 0

            # The actual input is the placeholder x reshaped to the input size of the network
            self.layer0_input = x[channel].reshape(self.layer0_input_size)
            self.layer0 = LeNetConvPoolLayer(rng,
                                        input=self.layer0_input,
                                        image_shape=self.layer0_input_size,
                                        subsample= (1,1),
                                        filter_shape=(num_kernels[0], 1) + kernel_sizes[0],
                                        poolsize=(2, 2))


            # ## Layer 1 - Second convolutional Layer
            # The second layer takes **`(batch_size, 10, 10, 10)`** as input, convolves it with 10 different **10x5x5** filters, and then downsamples (via maxpooling) in a **2x2** region.  Each filter/maxpool combination produces an output of size **`(10-5+1)/2 = 3`** on a side.
            # The size of the second layer's output is therefore **`(batch_size, 10, 3, 3)`**. 
            self.layer1_input_size = self.layer0_output_size
            self.edge1 = (self.edge0 - kernel_sizes[1][0] + 1) / 2
            self.layer1_output_size = (batch_size, num_kernels[1], self.edge1, self.edge1)

            # check that we have an even multiple of 2 before pooling
            assert ((self.edge0 - kernel_sizes[1][0] + 1) % 2) == 0

            self.layer1 = LeNetConvPoolLayer(rng,
                                        input=self.layer0.output,
                                        image_shape=self.layer1_input_size,
                                        subsample= (1,1),
                                        filter_shape=(num_kernels[1], num_kernels[0]) + kernel_sizes[1],
                                        poolsize=(2, 2))
                                        
    conv = Convolution(batch_size,num_kernels,kernel_sizes,0)
    conv2 = Convolution(batch_size,num_kernels,kernel_sizes,1)
    conv3 = Convolution(batch_size,num_kernels,kernel_sizes,2)
                                
    # ## Layer 2 - Fully connected sigmoidal layer
    #exit()
    # The sigmoidal layer takes a vector as input.
    # We flatten all but the first two dimensions, to get an input of size **`(batch_size, 30 * 4 * 4)`**.
    
    #raw_random= raw_random.RandomStreamsBase()
    srng = theano.tensor.shared_randomstreams.RandomStreams(
                        rng.randint(999999))
                        
    #def rectify(X):                                                         
    #    return T.maximum(X,0.) 
        
    def dropout(X,p=0.5):
        if p>0:
            retain_prob = 1-p
            X *= srng.binomial(X.shape,p=retain_prob,dtype = theano.config.floatX)
            X /= retain_prob
        return X
    
    def rectify(X): 
        return T.maximum(X,0.)

    layer2_input = conv.layer1.output.flatten(2)
    layer2_input = T.concatenate((T.concatenate((conv.layer1.output.flatten(2),conv2.layer1.output.flatten(2)),axis=1),conv2.layer1.output.flatten(2)),axis=1)

    layer2 = HiddenLayer(rng,
                         input=dropout(layer2_input),
                         n_in= num_kernels[1] * conv.edge1 * conv.edge1*3,
                         n_out= num_kernels[1] * conv.edge1 * conv.edge1,
                         activation=rectify) #T.tanh
                         
    # EXTRA LAYER
    # A fully connected logistic regression layer converts the sigmoid's layer output to a class label.
    extra =  HiddenLayer(rng,
                         input=dropout(layer2.output),
                         n_in= num_kernels[1] * conv.edge1 * conv.edge1,
                         n_out=num_kernels[1] * conv.edge1 * conv.edge1,
                         activation=rectify) #T.tanh


    # ## Layer 3 - Logistic regression output layer
    # A fully connected logistic regression layer converts the sigmoid's layer output to a class label.
    layer3 = LogisticRegression(input=extra.output,
                                n_in=num_kernels[1] * conv.edge1 * conv.edge1,
                                n_out=n_sports)
                                

    # # Training the network
    # To train the network, we have to define a cost function.  We'll use the Negative Log Likelihood of the model, relative to the true labels **`y`**.

    # The cost we minimize during training is the NLL of the model.
    # Recall: y is a placeholder we defined above
    cost = layer3.negative_log_likelihood(y)


    # ### Gradient descent
    # We will train with Stochastic Gradient Descent.  To do so, we need the gradient of the cost relative to the parameters of the model.  We can get the parameters for each label via the **`.params`** attribute.

    # create a list of all model parameters to be fit by gradient descent
    params = layer3.params + layer2.params + conv.layer1.params + conv.layer0.params

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    # ## Update
    updates = [
        (param_i, param_i - learning_rate * grad_i)  # <=== SGD update step
        for param_i, grad_i in zip(params, grads)
    ]

    index = T.lscalar()  # index to a batch of training/validation/testing examples

    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[:,index * batch_size: (index + 1) * batch_size],  # <=== batching
            y: train_set_y[index * batch_size: (index + 1) * batch_size]   # <=== batching
        }
    )

    # ## Validation function
    # To track progress on a held-out set, we count the number of misclassified examples in the validation set.
    validate_model = theano.function(
            [index],
            layer3.errors(y),
            givens={
                x: valid_set_x[:,index * batch_size: (index + 1) * batch_size],
                y: valid_set_y[index * batch_size: (index + 1) * batch_size]
            }
        )

    # ## Test function
    # After training, we check the number of misclassified examples in the test set.
    test_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            x: test_set_x[:,index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )   
	
    # guesses = theano.function(
    #         [],
    #         layer3.y_pred,
    #         givens={
    #             x: test_set_x
    #         }
    #     )
    # # Training loop 
    # We use SGD for a fixed number of iterations over the full training set (an "epoch").  Usually, we'd use a more complicated rule, such as iterating until a certain number of epochs fail to produce improvement in the validation set.  
    for epoch in range(90):
        costs = [train_model(i) for i in xrange(n_train_batches)]
        validation_losses = [validate_model(i) for i in xrange(n_valid_batches)]
        #print layer3.return_y_pred()
        print "Epoch {}    NLL {:.2}    %err in validation set {:.1%}".format(epoch + 1, np.mean(costs), np.mean(validation_losses))

    # ## Learned features
    #filters = tile_raster_images(layer0.W.get_value(borrow=True), img_shape=(9, 9), tile_shape=(1,10), tile_spacing=(3, 3),
    #                       scale_rows_to_unit_interval=True,
    #                       output_pixel_vals=True)

    #plt.imshow(filters)
    #plt.show()

    # ## Check performance on the test set
    test_errors = [test_model(i) for i in range(n_test_batches)]
    print "test errors: {:.1%}".format(np.mean(test_errors))

if __name__ == "__main__":
    run()

