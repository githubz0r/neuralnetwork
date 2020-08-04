import mlplol
import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets


diabetes = sklearn.datasets.load_diabetes()

def split_data_np(features, target, validation_fraction, test_fraction):
    N = features.shape[0]
    inds_shuffled = np.random.permutation(np.arange(0, N))
    train_thresh = int(np.round(inds_shuffled.shape[0] * (1 - test_fraction - validation_fraction)))
    val_thresh = int(np.round(inds_shuffled.shape[0] * (1 - test_fraction)))
    train_inds = inds_shuffled[0:train_thresh]
    val_inds = inds_shuffled[train_thresh:val_thresh]
    test_inds = inds_shuffled[val_thresh:]
    
    trainX = features[train_inds, :]
    trainY = target[train_inds]
    
    valX = features[val_inds, :]
    valY = target[val_inds]
    
    testX = features[test_inds, :]
    testY = target[test_inds]
    
    return dict(trainx=trainX, trainy=trainY, valx=valX, valy=valY, testx=testX, testy=testY)


diabetes_split = split_data_np(diabetes.data, diabetes.target, 0.2, 0.2)


print('''Before we train, lets check that the gradient is correct. We'll make up some random weights
    with e.g 10 neurons and check quotients of the gradient computed by the function and the finite difference grad.''')

D_test = 5
wtest1 = np.random.normal(0, 1, (diabetes.data.shape[1]+1, D_test))
wtest2 = np.random.normal(0, 1, (D_test+1, 1))

test_gradients = mlplol.gradient_quotients([wtest1, wtest2], diabetes_split['trainx'],
                diabetes_split['trainy'])
print(wtest1.shape, wtest2.shape)
print([i for i in test_gradients])


diabetes_nn = mlplol.NNregressor_onelayer(activation_function = 'relu')
diabetes_nn.estimate_weights(diabetes_split['trainx'], diabetes_split['trainy'], diabetes_split['valx'],
                             diabetes_split['valy'], n_hidden=100, 
                              iterations=100, patience=10, rate=0.001, 
                              verbose=False, weight_initialization_factors=None)

plt.style.use("dark_background")
plt.scatter(np.arange(diabetes_nn.iterations), diabetes_nn.training_loss, s=3, c='lime', label='Train')
plt.scatter(np.arange(diabetes_nn.iterations), diabetes_nn.validation_loss, s=3, c='fuchsia', label='Validation')
plt.ylabel('Loss')
plt.xlabel('Iterations')
plt.legend(loc='upper right')
plt.show()

test_pred = diabetes_nn.predict(diabetes_split['testx'])
test_loss = mlplol.squared_loss(test_pred, diabetes_split['testy'])
print('test loss: ', test_loss)