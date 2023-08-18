import numpy as np

# Function evaluating the sigmoid
def sigmoid(x):
    return 1/(1+np.exp(-x))

# Function evaluating the derivative of the sigmoid
def sigmoidPrime(x):
    return sigmoid(x)*(1-sigmoid(x))

# Function used to initialize the parameters of the neural network
def net_init(net):
    for i in range(1,len(net['struct'])):
        net['W'][i] = 0.01*np.random.normal(size=[net['struct'][i],net['struct'][i-1]])
        net['b'][i] = np.zeros([net['struct'][i],1])

# Function used to create the neural network structure. The input is a list
# of parameters corresponding to the length of the input, hidden and output
# layers. The output is a dictionary that contains all the parameters of the
# neuron network.
def net_create(st):
	net = {'struct': st, 'W': {}, 'b': {}, 'a':{}, 'h':{} }
	net_init(net)
	return net

# Function used to evaluate the neural network 
def net_predict(net,X):
    o = np.ones([1,X.shape[0]])
    
    net['h'][0] = np.transpose(X)
    for k in range(0,len(net['W'])):
        net['a'][k+1] = np.matmul(net['b'][k+1],o) + np.matmul(net['W'][k+1],net['h'][k])
        net['h'][k+1] = sigmoid(net['a'][k+1])
        
    return np.transpose(net['h'][len(net['W'])])

# Function evaluating the entropy loss
def net_loss(y,yhat):
    y = y.reshape([len(y),1])
    return np.sum(-(1-y)*np.log(1-yhat)-y*np.log(yhat))

# Function to determine the number of mismatches on the prediction
def net_missed(y,yhat):
    y = y.reshape([len(y),1])
    return np.sum(np.abs(y-(yhat>0.5)))

# Function using backpropagation to compute the gradients of the parameters
def net_backprop(net,X,y):
    # Performing forward propagation
    yhat = net_predict(net,X)
    
    # Initializing gradients
    nabla_b = {}
    nabla_W = {}
    
	# Implementation of gradients based on backpropagation algorithm

    G = yhat-y.reshape([len(y),1])                             # Initial value of G = derivative of L wrt to yhat
                                                               # First update of G is hadamard product of (g'(a(k))^T)
                                                               # and G
                                                               # G = (yhat - y) is the first update of G for last layer K

    for k in range(len(net['W']),0,-1):                        # Iterating from last layer back to first layer


        #For last layer (K), skip the first update of G as it is already taken care of in G = (yhat-y)


        nabla_b[k] = np.sum(G, axis=0)                         #Calculating partial derivative of L wrt b(k) =
                                                               # Sum of all entries in column of G

        nabla_W[k] = np.dot(net["h"][k-1],G)                   #Calculating partial derivative of L wrt W =
                                                               # h(k-1).G

        # Now calculating the value of G for the next layers -  G = hadamard product of (g'(a(k))^T) and G


        G_nabla_ak = np.multiply((sigmoidPrime(net["a"][k]).T),G)



        G= np.dot(G_nabla_ak,net["W"][k])                         #Updating G = G.W(k)
        
    return nabla_b, nabla_W

# Function used for training of the neural network. It updates the parameters
# in the neural network and returns the history of the training Loss 'Loss',
# the validation loss 'Loss_val' and the number of missed prediction in the
# validation set 'missed_val'.
def net_train(net,X_train,y_train,X_val,y_val,epsilon,NIter):
	# Initializing arrays holding the history of loss and missed values
    Loss = np.zeros(NIter)
    Loss_val = np.zeros(NIter)
    missed_val = np.zeros(NIter)
	
	# Simple implementation of gradient descent
    for n in range(0,NIter):
		# Computing gradient and updating parameters
        nabla_b, nabla_W = net_backprop(net,X_train,y_train)
        for k in range(0,len(net['W'])):
            net['b'][k+1] = net['b'][k+1] - epsilon*nabla_b[k+1].reshape(net['b'][k+1].shape)
            net['W'][k+1] = net['W'][k+1] - epsilon*np.transpose(nabla_W[k+1])
			
		# Computing losses and missed values for the validation set
        Loss[n] = net_loss(y_train,np.transpose(net['h'][len(net['W'])]))
        yhat_val = net_predict(net,X_val)
        Loss_val[n] = net_loss(y_val,yhat_val)
        missed_val[n] = net_missed(y_val,yhat_val)
		
		# Displaying results for the current epoch
        print("... Epoch {:3d} | Loss_Train: {:.2E} | Loss_Val: {:.2E} | Acc_Val: {:2.2f}".format(n,Loss[n],Loss_val[n],100-100*(missed_val[n])/len(yhat_val)))
	
    return Loss, Loss_val, missed_val