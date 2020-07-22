import numpy as np
from scipy import optimize
from .models import neural_network, nn_grad
import functools

def squared_loss(y1, y2):
    N = y1.shape[0]
    if len(y1.shape) < 2:
        y1 = np.expand_dims(y1, 1)
    if len(y2.shape) < 2:
        y2 = np.expand_dims(y2, 1)
    loss = 1*np.sum((y1 - y2)**2)/N
    return(loss)

def rebuild_from_shapes(flat_vector, shapes):
    lengths = [0] + [functools.reduce(lambda x,y: x*y, i) for i in shapes]
    subs = []
    for i in range(len(lengths)-1):
        index_1 = lengths[i]
        index_2 = lengths[i]+lengths[i+1]
        subs.append(flat_vector[index_1:index_2])
    for i,_ in enumerate(subs):
        subs[i] = subs[i].reshape(shapes[i])
    return(subs)

def nn_flat(w_flat, shapes, x, y, activation_function='relu', classify=False):
    w_subs = rebuild_from_shapes(w_flat, shapes)
    outputs = neural_network(x, w_subs, 'relu')['output']
    if not classify:
        error = 1*np.sum((outputs-y.reshape(y.shape[0], 1))**2)/y.shape[0]
    else:
        error = np.sum(y*np.log(outputs)+(1-y)*np.log(1-outputs))/y.shape[0]
    return(error)

def gradient_checker_(w, x, y, activation_function, eps, classify=False):
    '''using scipy approx prime, but must flatten weights and concatenate which
        kinda sucks'''
    if len(x.shape)<1:
        x = x.reshape(1,1)
    if len(y.shape)==1:
        y = y.reshape(y.shape[0], 1)
    shapes = [i.shape for i in w]
    nn_outs = neural_network(x, w, activation_function)
    a = nn_outs['first_mult']
    z = nn_outs['z1']
    out = nn_outs['output']
    x_ones = np.c_[np.ones(x.shape[0]), x]
    true_grad = nn_grad(x_ones, y, a, z, out, w, activation_function)
    flat_w = np.concatenate([i.flatten() for i in w])
    approx_grad = optimize.approx_fprime(flat_w, nn_flat, eps, shapes, x, y)
    approx_grad_split = rebuild_from_shapes(approx_grad, shapes)
    return(dict(true_grads = true_grad, approx_grads = approx_grad_split))

def calc_approx_grad(w, x, y, eps, true_loss, *args):
    finite_grads = [np.zeros(w_i.shape) for w_i in w]
    for k in range(len(w)):
        w_k = w[k]
        finite_grad_k = finite_grads[k]
        for i in np.arange(w_k.shape[0]):
            for j in np.arange(w_k.shape[1]):
                w_k_f = np.copy(w_k)
                w_k_f[i, j] = w_k[i, j]+eps
                w_list = w[0:k] + [w_k_f] + w[k+1:]
                output_kij = neural_network(x, w_list, *args)['output']
                finite_loss = squared_loss(y, output_kij)
                finite_diff = (finite_loss - true_loss)/eps
                finite_grad_k[i, j] = finite_diff
    return(finite_grads)



def gradient_checker(w, x, y, activation_function='relu', eps = np.sqrt(np.finfo(float).eps), 
                    classify=False):
    if len(x.shape)<1:
        x = x.reshape(1,1)
    if len(y.shape)==1:
        y = y.reshape(y.shape[0], 1)
    nn_outs = neural_network(x, w, activation_function)
    a = nn_outs['first_mult']
    z = nn_outs['z1']
    out = nn_outs['output']
    true_error = squared_loss(y, out)
    x_ones = np.c_[np.ones(x.shape[0]), x]
    true_grad = nn_grad(x_ones, y, a, z, out, w, activation_function)
    approx_grad = calc_approx_grad(w, x, y, eps, true_error, activation_function)
    return(dict(true_grads = true_grad, approx_grads = approx_grad))

def gradient_quotients(w, x, y, eps = np.sqrt(np.finfo(float).eps), 
                      activation_function='relu', classify=False):
    gradients = gradient_checker(w, x, y, activation_function, eps, classify)
    quotients = []
    for i,j in zip(gradients['true_grads'], gradients['approx_grads']):
        quotients.append((i+eps)/(j+eps))
    return(quotients)
