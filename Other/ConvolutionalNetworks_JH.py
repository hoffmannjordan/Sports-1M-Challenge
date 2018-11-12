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

sys.path.insert(1,'DeepLearningTutorials/code')
sys.path
import os

sport_n=7

class PreProcess(object):
    
    def __init__(self,load_in=False):
        self.load_in = load_in
        
    def run(self):

        if self.load_in == True:
            number = 600
			
            tmp = np.zeros((sport_n*number,100**2))
            j=1
            for i in xrange(1,number+1):
                file=np.genfromtxt('../Beach_Soccer/BS_'+str(i)+'.csv',delimiter=',',dtype=None)
                file = file.reshape((1,100**2))
                tmp[number*(j-1)+i-1] = file
            j=2
            for i in xrange(1,number+1):
                file=np.genfromtxt('../Bowling/BO_'+str(i)+'.csv',delimiter=',',dtype=None)
                file = file.reshape((1,100**2))
                tmp[number*(j-1)+i-1] =file
            j=3
            for i in xrange(1,number+1):
                file=np.genfromtxt('../Karate/KA_'+str(i)+'.csv',delimiter=',',dtype=None)
                file = file.reshape((1,100**2))
                tmp[number*(j-1)+i-1] = file
            j=4
            for i in xrange(1,number+1):
                file=np.genfromtxt('../Rugby/RUGBY_'+str(i)+'.csv',delimiter=',',dtype=None)
                file = file.reshape((1,100**2))
                tmp[number*(j-1)+i-1] = file
            j=5
            for i in xrange(1,number+1):
                file=np.genfromtxt('../Kayaking/KAYAK_'+str(i)+'.csv',delimiter=',',dtype=None)
                file = file.reshape((1,100**2))
                tmp[number*(j-1)+i-1] = file
            j=6
            for i in xrange(1,number+1):
                file=np.genfromtxt('../Hurdles/HU_'+str(i)+'.csv',delimiter=',',dtype=None)
                file = file.reshape((1,100**2))
                tmp[number*(j-1)+i-1] = file
            j=7
            for i in xrange(1,number+1):
                file=np.genfromtxt('../Surfing/Surf_'+str(i)+'.csv',delimiter=',',dtype=None)
                file = file.reshape((1,100**2))
                tmp[number*(j-1)+i-1] = file				

            tmp_y = np.zeros(tmp.shape[0])
            for n in xrange(tmp_y.size):
                tmp_y[n] = np.floor(n/float(number))
        
            np.save('tmp.npy',tmp)
            np.save('tmp_y.npy',tmp_y)
    
        else:
            tmp = np.load('tmp.npy')
            tmp_y = np.load('tmp_y.npy')


        mod_n=3

        train_x= np.zeros((sport_n*number/mod_n*(mod_n-1),100**2))
        test_x= np.zeros((sport_n*number/mod_n,100**2))

        train_y = np.zeros(sport_n*number/mod_n*(mod_n-1))
        test_y = np.zeros(sport_n*number/mod_n)

        count_train = 0
        count_test = 0
        for n in xrange(tmp_y.size):
            if n%mod_n==0:
                test_x[count_test] = tmp[n]
                test_y[count_test] = tmp_y[n]
                count_test += 1
    
            else:
                train_x[count_train] = tmp[n]
                train_y[count_train] = tmp_y[n]
                count_train += 1

        train_set_x, train_set_y = train_x,train_y
        vs = 300 #valid set size
        a = np.random.permutation(range(train_x.shape[0]))[0:vs]
        print a

        valid_set_x = np.zeros((vs,train_set_x.shape[1]))
        valid_set_y = np.zeros(vs)
        for n in xrange(len(a)):
            valid_set_x[n] = train_set_x[a[n]]
            valid_set_y[n] = train_set_y[a[n]]
    
        test_set_x, test_set_y = test_x,test_y

        train_set_x = train_set_x.astype(np.float32)
        test_set_x = test_set_x.astype(np.float32)
        valid_set_x = valid_set_x.astype(np.float32)

        train_set_y = train_set_y.astype(np.int32)
        test_set_y = test_set_y.astype(np.int32)
        valid_set_y = valid_set_y.astype(np.int32)

        # estimate the mean and std dev from the training data
        # then use these estimates to normalize the data
        # estimate the mean and std dev from the training data

        norm_mean = train_set_x.mean(axis=0)
        train_set_x = train_set_x - norm_mean
        norm_std = train_set_x.std(axis=0)
        norm_std = norm_std.clip(0.00001, norm_std.max())
        train_set_x = train_set_x / norm_std 

        test_set_x = test_set_x - norm_mean
        test_set_x = test_set_x / norm_std 
        valid_set_x = valid_set_x - norm_mean
        valid_set_x = valid_set_x / norm_std 

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
    
    preProcess = PreProcess(load_in=True)
    data = preProcess.run()
    
    train_set_x,train_set_y = data[0],data[3]
    valid_set_x,valid_set_y = data[1],data[4]
    test_set_x,test_set_y = data[2],data[5]
    
    # network parameters
    num_kernels = [10, 10]
    kernel_sizes = [(9, 9), (5, 5)]
    #exit()
    sigmoidal_output_size = 20

    # training parameters
    learning_rate = 0.1
    batch_size = 50

    # Setup 2: compute batch sizes for train/test/validation
    # borrow=True gets us the value of the variable without making a copy.
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_train_batches /= batch_size
    n_test_batches /= batch_size
    n_valid_batches /= batch_size


    # Setup 3.
    # Declare inputs to network - x and y are placeholders
    # that will be used in the training/testing/validation functions below.
    x = T.matrix('x')  # input image data
    y = T.ivector('y') # input label data

    # ## Layer 0 - First convolutional Layer
    # The first layer takes **`(batch_size, 1, 28, 28)`** as input, convolves it with **10** different **9x9** filters, and then downsamples (via maxpooling) in a **2x2** region.  Each filter/maxpool combination produces an output of size **`(28-9+1)/2 = 10`** on a side.
    # The size of the first layer's output is therefore **`(batch_size, 10, 10, 10)`**. 

    layer0_input_size = (batch_size, 1, 100, 100)  # fixed size from the data
    edge0 = (100 - kernel_sizes[0][0] + 1) / 2
    layer0_output_size = (batch_size, num_kernels[0], edge0, edge0)
    # check that we have an even multiple of 2 before pooling
    assert ((100 - kernel_sizes[0][0] + 1) % 2) == 0

    # The actual input is the placeholder x reshaped to the input size of the network
    layer0_input = x.reshape(layer0_input_size)
    layer0 = LeNetConvPoolLayer(rng,
                                input=layer0_input,
                                image_shape=layer0_input_size,
                                filter_shape=(num_kernels[0], 1) + kernel_sizes[0],
                                poolsize=(2, 2))


    # ## Layer 1 - Second convolutional Layer
    # The second layer takes **`(batch_size, 10, 10, 10)`** as input, convolves it with 10 different **10x5x5** filters, and then downsamples (via maxpooling) in a **2x2** region.  Each filter/maxpool combination produces an output of size **`(10-5+1)/2 = 3`** on a side.
    # The size of the second layer's output is therefore **`(batch_size, 10, 3, 3)`**. 
    layer1_input_size = layer0_output_size
    edge1 = (edge0 - kernel_sizes[1][0] + 1) / 2
    layer1_output_size = (batch_size, num_kernels[1], edge1, edge1)
    # check that we have an even multiple of 2 before pooling
    assert ((edge0 - kernel_sizes[1][0] + 1) % 2) == 0

    layer1 = LeNetConvPoolLayer(rng,
                                input=layer0.output,
                                image_shape=layer1_input_size,
                                filter_shape=(num_kernels[1], num_kernels[0]) + kernel_sizes[1],
                                poolsize=(2, 2))


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
        


    layer2_input = layer1.output.flatten(2)

    layer2 = HiddenLayer(rng,
                         input=dropout(layer2_input),
                         n_in= num_kernels[1] * edge1 * edge1,
                         n_out=sigmoidal_output_size,
                         activation=T.tanh)


    # ## Layer 3 - Logistic regression output layer
    # A fully connected logistic regression layer converts the sigmoid's layer output to a class label.
    layer3 = LogisticRegression(input=layer2.output,
                                n_in=sigmoidal_output_size,
                                n_out=sport_n)

    # # Training the network
    # To train the network, we have to define a cost function.  We'll use the Negative Log Likelihood of the model, relative to the true labels **`y`**.

    # The cost we minimize during training is the NLL of the model.
    # Recall: y is a placeholder we defined above
    cost = layer3.negative_log_likelihood(y)


    # ### Gradient descent
    # We will train with Stochastic Gradient Descent.  To do so, we need the gradient of the cost relative to the parameters of the model.  We can get the parameters for each label via the **`.params`** attribute.

    # create a list of all model parameters to be fit by gradient descent
    params = layer3.params + layer2.params + layer1.params + layer0.params

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
            x: train_set_x[index * batch_size: (index + 1) * batch_size],  # <=== batching
            y: train_set_y[index * batch_size: (index + 1) * batch_size]   # <=== batching
        }
    )

    # ## Validation function
    # To track progress on a held-out set, we count the number of misclassified examples in the validation set.
    validate_model = theano.function(
            [index],
            layer3.errors(y),
            givens={
                x: valid_set_x[index * batch_size: (index + 1) * batch_size],
                y: valid_set_y[index * batch_size: (index + 1) * batch_size]
            }
        )

    # ## Test function
    # After training, we check the number of misclassified examples in the test set.
    test_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # # Training loop 
    # We use SGD for a fixed number of iterations over the full training set (an "epoch").  Usually, we'd use a more complicated rule, such as iterating until a certain number of epochs fail to produce improvement in the validation set.  
    for epoch in range(30):
        costs = [train_model(i) for i in xrange(n_train_batches)]
        validation_losses = [validate_model(i) for i in xrange(n_valid_batches)]
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

