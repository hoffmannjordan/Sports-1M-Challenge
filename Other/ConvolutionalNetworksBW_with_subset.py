import sys
import time as time
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import theano
import theano.tensor as T

matplotlib.pyplot.gray()
theano.config.floatX = 'float32'
rng = np.random.RandomState(42)

from convolutional_mlp import LeNetConvPoolLayer
from mlp               import HiddenLayer
from logistic_sgd      import LogisticRegression
from pre_process       import PreProcess

# THEANO_FLAGS= 'openmp=True' python ConvolutionalNetworks.py 
        
class Convolution(object):
    '''
    Class that defines the hierarchy and design of the convolutional
    layers.
    '''
    
    def __init__(self,batch_size,num_kernels,kernel_sizes,channel,x,y):
        self.layer0_input_size  = (batch_size, 1, 100, 100)                             # Input size from data 
        self.edge0              = (100 - kernel_sizes[0][0] + 1) / 3                    # New edge size
        self.layer0_output_size = (batch_size, num_kernels[0], self.edge0, self.edge0)  # Output size
        assert ((100 - kernel_sizes[0][0] + 1) % 3) == 0                                # Check pooling size
        
        # Initialize Layer 0
        self.layer0_input = x.reshape(self.layer0_input_size)
        self.layer0 = LeNetConvPoolLayer(rng,
                                    input=self.layer0_input,
                                    image_shape=self.layer0_input_size,
                                    subsample= (1,1),
                                    filter_shape=(num_kernels[0], 1) + kernel_sizes[0],
                                    poolsize=(3, 3))

        self.layer1_input_size  = self.layer0_output_size                              # Input size Layer 1
        self.edge1              = (self.edge0 - kernel_sizes[1][0] + 1) / 2            # New edge size
        self.layer1_output_size = (batch_size, num_kernels[1], self.edge1, self.edge1) # Output size
        assert ((self.edge0 - kernel_sizes[1][0] + 1) % 2) == 0                        # Check pooling size

        # Initialize Layer 1
        self.layer1 = LeNetConvPoolLayer(rng,
                                    input=self.layer0.output,
                                    image_shape=self.layer1_input_size,
                                    subsample= (1,1),
                                    filter_shape=(num_kernels[1], num_kernels[0]) + kernel_sizes[1],
                                    poolsize=(2, 2))
                                    

class Functions(object):
    '''
    Class containing helper functions for the ConvNet class.
    '''
    
    def dropout(self,X,p=0.5):
        '''
        Perform dropout with probability p
        '''
        if p>0:
            retain_prob = 1-p
            X *= self.srng.binomial(X.shape,p=retain_prob,dtype = theano.config.floatX)
            X /= retain_prob
        return X
        
    def vstack(self,layers):
        '''
        Vstack
        '''
        n = 0
        for layer in layers:
            if n == 1:
                out_layer = T.concatenate(layer,layers[n-1])
            elif n>1:
                out_layer = T.concatenate(out_layer,layer)
            n += 1
        return out_layer

    def rectify(self,X): 
        '''
        Rectified linear activation function
        '''
        return T.maximum(X,0.)
        
    def RMSprop(self,cost, params, lr = 0.001, rho=0.9, epsilon=1e-6):
        '''
        RMSprop - NEED MORE INFORMATION ABOUT THIS
        '''
        grads = T.grad(cost=cost, wrt=params)
        updates = []
        for p, g in zip(params, grads):
            acc              = theano.shared(p.get_value() * 0.)
            acc_new          = rho * acc + (1 - rho) * g ** 2
            gradient_scaling = T.sqrt(acc_new + epsilon)
            g                = g / gradient_scaling
            
            updates.append((acc, acc_new))
            updates.append((p, p - lr * g))
            
        return updates
    
    def stochasticGradient(self,cost,params,lr):
        '''
        Stochastic Gradient Descent
        '''
        updates = [
            (param_i, param_i - lr * grad_i)  # <=== SGD update step
            for param_i, grad_i in zip(params, grads)
        ]
        return updates
    
    def gradient_updates_momentum(self,cost, params, learning_rate, momentum):
        '''
        Compute updates for gradient descent with momentum
    
        :parameters:
            - cost : theano.tensor.var.TensorVariable
                Theano cost function to minimize
            - params : list of theano.tensor.var.TensorVariable
                Parameters to compute gradient against
            - learning_rate : float
                Gradient descent learning rate
            - momentum : float
                Momentum parameter, should be at least 0 (standard gradient descent) and less than 1
   
        :returns:
            updates : list
                List of updates, one for each parameter
        '''
        # Make sure momentum is a sane value
        assert momentum < 1 and momentum >= 0
        # List of update steps for each parameter
        updates = []
        # Just gradient descent on cost
        for param in params:
            # For each parameter, we'll create a param_update shared variable.
            # This variable will keep track of the parameter's update step across iterations.
            # We initialize it to 0
            param_update = theano.shared(param.get_value()*0., broadcastable=param.broadcastable)
            # Each parameter is updated by taking a step in the direction of the gradient.
            # However, we also "mix in" the previous step according to the given momentum value.
            # Note that when updating param_update, we are using its old value and also the new gradient step.
            updates.append((param, param - learning_rate*param_update))
            # Note that we don't need to derive backpropagation to compute updates - just use T.grad!
            updates.append((param_update, momentum*param_update + (1. - momentum)*T.grad(cost, param)))
        return updates

        
        
    def init_optimizer(self, optimizer, cost, params, optimizerData):
        '''
        Choose between different optimizers 
        '''
        if optimizer == 'stochasticGradient':
            updates = self.stochasticGradient(cost, 
                                              params,
                                              lr      = optimizerData['learning_rate'])
        elif optimizer == 'RMSprop':    
            updates = self.RMSprop(cost, params, optimizerData['learning_rate'],
                                                 rho     = optimizerData['rho'],
                                                 epsilon = optimizerData['epsilon'])
        elif optimizer == 'momentum':
            updates = self.gradient_updates_momentum(cost, 
                                                    params, 
                                                    optimizerData['learning_rate'], 
                                                    optimizerData['momentum'])
        return updates


class ConvNet(Functions):
    '''
    Main function for the convolutional network
    '''
    
    def __init__(self):
        
        self.srng = theano.tensor.shared_randomstreams.RandomStreams(
                            rng.randint(999999))
                            
    
    def stack(self,stackitems):
        for n in xrange(len(stackitems)-1):
            if n == 0:
                output = T.concatenate((stackitems[n],stackitems[n+1]),axis=1)
            else:
                output = T.concatenate((output,stackitems[n+1]),axis=1)
        return output
                            
    
    def model(self,batch_size,num_kernels,kernel_sizes,x,y):
            
        # convolutional layers
        conv  = Convolution(batch_size, num_kernels, kernel_sizes, 0, x, y)
        #conv2 = Convolution(batch_size, num_kernels, kernel_sizes, 1, x, y)
        #conv3 = Convolution(batch_size, num_kernels, kernel_sizes, 2, x, y)
                    

        layer2_input = conv.layer1.output.flatten(2)
        #stacklist = [conv.layer1.output.flatten(2),conv2.layer1.output.flatten(2),conv3.layer1.output.flatten(2)]
        #layer2_input = self.stack(stacklist)

        layer2 = HiddenLayer(rng,
                                  input      = self.dropout(layer2_input),
                                  n_in       = num_kernels[1] * conv.edge1 * conv.edge1,
                                  n_out      = num_kernels[1] * conv.edge1 * conv.edge1,
                                  activation = self.rectify)


        # ## Layer 3 - Logistic regression output layer
        layer3 = LogisticRegression(input = self.dropout(layer2.output),
                                         n_in  = num_kernels[1] * conv.edge1 * conv.edge1,
                                         n_out = self.n_sports)
        
        convparams = conv.layer1.params + conv.layer0.params #+ conv2.layer1.params + conv2.layer0.params + conv3.layer1.params + conv3.layer0.params
        hiddenparams = layer3.params + layer2.params
        self.params = hiddenparams + convparams 
        self.conv   = conv
        #self.conv2  = conv2
        #self.conv3  = conv3
        self.layer2 = layer2
        self.layer3 = layer3
                                         

    def run(self,
            num_kernels  = [15,15],
            kernel_sizes = [(11, 11), (5, 5)],
            batch_size   = 50,
            epochs       = 20,
            optimizer    = 'RMSprop'):
            
            
        optimizerData = {}
        optimizerData['learning_rate'] = 0.001
        optimizerData['rho']           = 0.9
        optimizerData['epsilon']       = 1e-4
        optimizerData['momentum']      = 0.9
        
        print '... Loading data'
        
        # load in and process data
        preProcess              = PreProcess()
        data                    = preProcess.run()
        train_set_x,train_set_y = data[0],data[3]
        valid_set_x,valid_set_y = data[1],data[4]
        test_set_x,test_set_y   = data[2],data[5]

        print train_set_x.eval().shape

        print '... Initializing network'
       
        # training parameters
        self.n_sports = np.max(train_set_y.eval())+1
    
        # print error if batch size is to large
        if valid_set_y.eval().size<batch_size:
            print 'Error: Batch size is larger than size of validation set.'

        # compute batch sizes for train/test/validation
        n_train_batches  = train_set_x.get_value(borrow=True).shape[0]
        n_test_batches   = test_set_x.get_value(borrow=True).shape[0]
        n_valid_batches  = valid_set_x.get_value(borrow=True).shape[0]
        n_train_batches /= batch_size
        n_test_batches  /= batch_size
        n_valid_batches /= batch_size

        # symbolic variables
        x = T.matrix('x')  # input image data
        y = T.ivector('y')  # input label data
        
        self.model(batch_size, num_kernels, kernel_sizes, x, y)

        # Initialize parameters and functions
        cost   = self.layer3.negative_log_likelihood(y)        # Cost function
        params = self.params                                   # List of parameters
        grads  = T.grad(cost, params)                          # Gradient
        index  = T.lscalar()                                   # Index
        
        # Intialize optimizer
        updates = self.init_optimizer(optimizer, cost, params, optimizerData)

        # Training model
        train_model = theano.function(
                      [index],
                      cost,
                      updates = updates,
                      givens  = {
                                x: train_set_x[index * batch_size: (index + 1) * batch_size], 
                                y: train_set_y[index * batch_size: (index + 1) * batch_size] 
            }
        )

        # Validation function
        validate_model = theano.function(
                         [index],
                         self.layer3.errors(y),
                         givens = {
                                  x: valid_set_x[index * batch_size: (index + 1) * batch_size],
                                  y: valid_set_y[index * batch_size: (index + 1) * batch_size]
                }
            )

        # Test function
        test_model = theano.function(
                     [index],
                     self.layer3.errors(y),
                     givens = {
                              x: test_set_x[index * batch_size: (index + 1) * batch_size],
                              y: test_set_y[index * batch_size: (index + 1) * batch_size]
            }
        )
        def solve():
            costs = []
            for i in xrange(n_train_batches):
                costs.append(train_model(i))

                #if i % 1000 ==0:
                #    print i
            return costs
        # Solver
        try:
            print '... Solving'
            start_time = time.time()    
            for epoch in range(epochs):
                t1 = time.time()
                costs             = solve()
                validation_losses = [validate_model(i) for i in xrange(n_valid_batches)]
                t2 = time.time()
                print "Epoch {}    NLL {:.2}    %err in validation set {:.1%}    Time (epoch/total) {:.2}/{:.2} mins".format(epoch + 1, np.mean(costs), np.mean(validation_losses),(t2-t1)/60.,(t2-start_time)/60.)
        except KeyboardInterrupt:
            print '... Exiting solver'
        # Evaluate performance 
        test_errors = [test_model(i) for i in range(n_test_batches)]
        print "test errors: {:.1%}".format(np.mean(test_errors))

if __name__ == "__main__":
    convnet = ConvNet()
    convnet.run()
    
    
