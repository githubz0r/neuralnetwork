import numpy as np

def neural_network(x, w, activation_function, classify = False, has_ones = False):
    w0 = w[0]
    w1 = w[1]
    if len(x.shape)<1:
        x = x.reshape(1,1)
    N = x.shape[0]
    if not has_ones:
        x = np.c_[np.ones(N), x]
    a = x@w0 # dimenson is d*M where M is number of columns in w0, i.e. number of neurons
    if activation_function == 'relu':
        z = relu(a)
    elif activation_function == 'softsign':
        z = a/(1+np.abs(a))
    z = np.c_[np.ones(z.shape[0]), z] # ones for bias
    a2 = z@w1
    if not classify:
        z2 = a2
    else:
        z2 = np.exp(z)/(np.exp(z)+1)
    return(dict(first_mult = a, first_mult_nonlin = z, second_mult = a2, output = z2))

def relu(x):
    zeroes = np.zeros(x.shape)
    zeroes[x>0] = x[x>0]
    return(zeroes)

def relu_grad(x):
    grad = (x > 0)*1
    return(grad)


def nn_grad(x, y, a, z, out, w, activation_function):
    '''y must be an array of dimension at least 2'''
    w_out = w[1]
    N = x.shape[0]
    delta_outs = out-y # N*1
    #delta_outs_repeated = np.repeat((out-y), w_out.shape[0], axis=1) # N*(n_hidden+1)
    #output_grad1 = np.zeros(w_out.shape)
    #for i in range(N):
        #w_i = np.outer(z[i, :], delta_outs[i, :])
        #output_grad1 += w_i/N
    output_grad = np.tensordot(z, delta_outs, axes=([0, 0]))/N
    #output_grad = np.sum(np.multiply(delta_outs, z), axis=0)/N # gradient of output unit
    if len(output_grad.shape)<2:
        output_grad = output_grad.reshape(output_grad.shape[0], 1)
    if activation_function == 'relu':
        hidden_activation_deriv = relu_grad(a)
    elif activation_function == 'softsign':
        hidden_activation_deriv = 1/((1+np.abs(a))**2)
    #delta_hidden_sum_parts = delta_outs@w_out.T # this must be a sum with more than 1 output neuron
    delta_hidden_sum_parts = (w_out@delta_outs.T).T
    delta_hidden_sum_parts = delta_hidden_sum_parts[:,1:] # removing bias column
    delta_hidden = hidden_activation_deriv*delta_hidden_sum_parts
    hidden_grad = 0
    for i in range(N):
        vector_of_deltas = delta_hidden[i,:]
        vector_of_deltas = vector_of_deltas.reshape(a.shape[1],1)
        grad_element = x[i,:]*vector_of_deltas
        hidden_grad += grad_element/N
    return([2*hidden_grad.T, 2*output_grad])

def nn_gradient_descent(x_train, y_train, x_val, y_val, n_hidden, rate, iterations, patience,
                       verbose, weights, initialization_factors, activation_function):
    if len(y_train.shape)==1:
        y_train = y_train.reshape(y_train.shape[0], 1)
    if len(y_val.shape)==1:
        y_val = y_val.reshape(y_val.shape[0], 1)
    if len(x_train.shape)==1:
        x_train = x_train.reshape(x_train.shape[0], 1)
    K = y_train.shape[1]
    D = x_train.shape[1]
    N = x_train.shape[0]
    x_train = np.c_[np.ones(N), x_train, ]
    # initialize weights
    if initialization_factors == None:
        initialization_factors = [np.sqrt(2/(D+1 + n_hidden)), np.sqrt(2/(n_hidden+1 + K))]
    if weights == None:
        w_hidden0 = np.random.normal(0, 1, (D+1, n_hidden)) * initialization_factors[0]
        w_out0 = np.random.normal(0, 1, (n_hidden+1, K)) * initialization_factors[1]
    w_hidden1 = None
    w_out1 = None
    w_best = None
    
    patience_counter = patience
    train_loss = []
    val_loss = []
    gradient_norm = []
    max_iter = iterations
    while iterations > 0 and patience_counter >0:
        w=[w_hidden0, w_out0]
        nn_outs_train = neural_network(x_train, w, activation_function, has_ones=True)
        a = nn_outs_train['first_mult']
        z = nn_outs_train['first_mult_nonlin']
        out = nn_outs_train['output']
        train_error = 1*np.sum((y_train - out)**2)/N
        if len(out.shape)<2:
            out=out.reshape(y_train.shape[0], 1) # reshaping to get broadcasting to work later
        train_loss.append(train_error)
        grads = nn_grad(x=x_train, y=y_train, a=a, z=z, out=out, w=w, activation_function=activation_function)
        hidden_grad = grads[0]
        output_grad = grads[1]
        w_hidden1 = w_hidden0 - rate*hidden_grad #/N
        w_out1 = w_out0 - rate*output_grad #/N
        val_out = neural_network(x_val, w, activation_function=activation_function)['output']
        val_error = 1*np.sum((y_val - val_out)**2)/x_val.shape[0]
        iterations -= 1
        if verbose: 
            print('iterations: ', max_iter-iterations)
            print('Training loss: {}, Validation loss: {}'.format(train_error, val_error))
        if len(val_loss)>1:
            if val_error < min(val_loss):
                if verbose: 
                    print('new best w')
                w_best = w
                patience_counter = patience
            elif val_error >= val_loss[-1]:
                patience_counter -= 1
        val_loss.append(val_error)
        w_out0 = w_out1
        w_hidden0 = w_hidden1
        gradient_norm_i = np.sqrt(np.sum(np.concatenate([hidden_grad.flatten(), output_grad.flatten()])**2))
        gradient_norm.append(gradient_norm_i)
    if w_best == None:
        w_best = [w_hidden0, w_out0]
    return(dict(weights=w_best, train_loss=train_loss, val_loss=val_loss, gradient_norm=gradient_norm,
               iterations=max_iter-iterations))

class NNregressor_onelayer:
    def __init__(self, activation_function, weights=None):
        self.weights = weights
        self.activation_function = activation_function
    def estimate_weights(self, trainx, trainy, valx, valy, n_hidden, rate, iterations, patience, verbose,
                        weight_initialization_factors):
        training_results = nn_gradient_descent(trainx, trainy, valx, valy, n_hidden, 
                                           rate, iterations, patience, verbose,
                                           self.weights, weight_initialization_factors,
                                               activation_function = self.activation_function)
        self.weights = training_results['weights']
        self.training_loss = training_results['train_loss']
        self.validation_loss = training_results['val_loss']
        self.gradient_norm = training_results['gradient_norm']
        self.iterations = training_results['iterations']
    def predict(self, x):
        predictions = neural_network(x, self.weights, activation_function = self.activation_function)
        if len(x.shape) < 2:
            pred = predictions['output'].ravel()
        else:
            pred = predictions['output']
        return(pred)
