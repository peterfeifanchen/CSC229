#!/usr/bin/python
from __future__ import division
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import Q1

def vecmatmul( b_T, A ):
    col = A.shape[1]
    ret = np.zeros((1, col))

    for i in xrange(0, col):
        ret[0][i] = b_T[:].dot(A[:,i])

    return ret

def matvecmul( A, b ):
    row = A.shape[0]
    ret = np.zeros((row, 1))

    for i in xrange(0, row):
        ret[i][0] = A[i,:].dot(b[:])

    return ret

def matmatmul( A, B ):
    row = A.shape[0]
    col = B.shape[1]
    ret = np.zeros((row, col))

    for i in xrange(0, row):
        for j in xrange(0, col):
            ret[i][j] = A[i,:].dot(B[:,j])

    return ret

def load_data():
    train = np.genfromtxt('quasar_train.csv', skip_header=True, delimiter=',')
    test = np.genfromtxt('quasar_test.csv', skip_header=True, delimiter=',')
    wavelengths = np.genfromtxt('quasar_train.csv', skip_header=False, delimiter=',')[0]
    return train, test, wavelengths

def add_intercept(X_):
    return np.column_stack( (X_, np.array( [1]*450 )) )

def smooth_data(Y, X_, tau):
    sample = 0
    ysmooth = np.zeros(Y.shape)
    for i in xrange(0, Y.shape[0]):
        sample = sample + 1
        print( 'sample {}'.format(sample) )
        ysmooth[i,:] = LWR_smooth( Y[i], X_, tau )
    return ysmooth

def LWR_smooth(Y, X_, tau):
    yhat = np.zeros(Y.shape)
    ###############
    X = add_intercept(X_)
    W = np.identity(X.shape[0])
    for j in xrange(0, X.shape[0]):
        for i in xrange(0, X.shape[0]):
            W[i][i] = np.exp(-(X_[i]-X_[j])*(X_[i]-X_[j])/(2*tau*tau))
        X_TW = matmatmul( X.transpose(), W )
        X_TWX_1 = np.linalg.inv( matmatmul( X_TW, X ) )
        X_TWX_1X_TW = matmatmul( X_TWX_1, X_TW )
        theta_ = matvecmul( X_TWX_1X_TW, Y )
        yhat[j] = theta_[:,0].dot(X[j,:])

    ###############
    return yhat

def LR_smooth(Y, X_):
    X = add_intercept(X_)
    yhat = np.zeros(Y.shape)
    X_TX_1 = np.linalg.inv( matmatmul( X.transpose(), X ) )
    X_TX_1X_T = matmatmul( X_TX_1, X.transpose() )
    theta = matvecmul( X_TX_1X_T, Y )
    yhat = matvecmul( X, theta )

    return yhat, theta

def plot_b(X, raw_Y, Ys, desc, filename):
    plt.figure()
    plt.scatter(X, raw_Y, c='red')
    
    for i in xrange(0, len(Ys)):
        plt.plot(X, Ys[i], label=desc[i])
    plt.legend()
    plt.show()

    plt.savefig(filename)

def plot_c(Yhat, Y, X, filename):
    plt.figure()
    plt.plot(X, Y)
    plt.plot(X[0:len(Yhat)], Yhat)
    plt.show()

    plt.savefig(filename)
    return

def split(full, wavelengths):
    leftSize = 0
    rightSize = 0
    for i in xrange( 0, len(wavelengths) ):
        if wavelengths[i] < 1200:
            leftSize += 1
        if wavelengths[i] >= 1300:
            rightSize += 1
    left = np.zeros( ( full.shape[ 0 ], leftSize ) )
    right = np.zeros( ( full.shape[ 0 ], rightSize ) )
    ###############
    for i in xrange( 0, full.shape[ 0 ] ):
        left[i,:] = [ full[i][j] for j in xrange( 0, len(wavelengths) ) 
                                if wavelengths[j] < 1200 ]
        right[i, :] = [ full[i][j] for j in xrange( 0, len(wavelengths) ) 
                                if wavelengths[j] >= 1300 ]
    ###############
    return left, right

def dist(a, b):
    dist = 0
    ################
    dist = (a-b).dot((a-b))
    ################
    return dist

def knearest3( right_train, right_test ):
    nearestD = [0, 0, 0]
    nearestI = [-1, -1, -1]
    maxD = 0

    for i in xrange( 0, len(right_train) ): 
        d = dist( right_train[i], right_test )
        if d > maxD:
            maxD = d
        for j in xrange( 0, 3 ):
            if nearestI[ j ] == -1:
                nearestD[ j ] = d
                nearestI[ j ] = i
                break
            if d < nearestD[ j ]:
                for k in xrange( 2, j, -1 ):
                    nearestD[k] = nearestD[k-1]
                    nearestI[k] = nearestI[k-1]
                nearestD[j] = d
                nearestI[j] = i
                break
        
    return maxD, nearestD, nearestI

def func_reg(left_train, right_train, right_test):
    m, n = left_train.shape
    ###########################
    maxD, nearestD, nearestI = knearest3( right_train, right_test )
    sumKer = 0
    sumLeft = np.zeros(left_train[0].shape[0],)
    for i in xrange(0,len(nearestD)):
        ker = max( 1 - nearestD[i]/maxD, 0 )
        sumLeft += ker*left_train[nearestI[i]]
        sumKer += ker
    ###########################
    return sumLeft / sumKer

def main():
    raw_train, raw_test, wavelengths = load_data()

    ## Part b.i
    # lr_est, theta = LR_smooth(raw_train[0], wavelengths)
    # print('Part b.i) Theta=[%.4f, %.4f]' % (theta[0], theta[1]))
    # plot_b(wavelengths, raw_train[0], [lr_est], ['Regression line'], 'p5b1.png')

    ## Part b.ii
    # lwr_est_5 = LWR_smooth(raw_train[0], wavelengths, 5)
    # plot_b(wavelengths, raw_train[0], [lwr_est_5], ['tau = 5'], 'p5b2.png')

    ### Part b.iii
    # lwr_est_1 = LWR_smooth(raw_train[0], wavelengths, 1)
    # lwr_est_10 = LWR_smooth(raw_train[0], wavelengths, 10)
    # lwr_est_100 = LWR_smooth(raw_train[0], wavelengths, 100)
    # lwr_est_1000 = LWR_smooth(raw_train[0], wavelengths, 1000)
    # plot_b(wavelengths, raw_train[0],
    #        [lwr_est_1, lwr_est_5, lwr_est_10, lwr_est_100, lwr_est_1000],
    #        ['tau = 1', 'tau = 5', 'tau = 10', 'tau = 100', 'tau = 1000'],
    #        'p5b3.png')

    ### Part c.i
    smooth_train, smooth_test = [smooth_data(raw, wavelengths, 5) for 
            raw in [raw_train, raw_test]]

    #### Part c.ii
    left_train, right_train = split(smooth_train, wavelengths)
    left_test, right_test = split(smooth_test, wavelengths)

    train_errors = [dist(left, func_reg(left_train, right_train, right)) for 
                           left, right in zip(left_train, right_train)]
    print('Part c.ii) Training error: %.4f' % np.mean(train_errors))

    ### Part c.iii
    test_errors = [dist(left, func_reg(left_train, right_train, right)) 
                        for left, right in zip(left_test, right_test)]
    print('Part c.iii) Test error: %.4f' % np.mean(test_errors))

    left_1 = func_reg(left_train, right_train, right_test[0])
    plot_c(left_1, smooth_test[0], wavelengths, 'p5c3-1.png')
    left_6 = func_reg(left_train, right_train, right_test[5])
    plot_c(left_6, smooth_test[5], wavelengths, 'p5c3-6.png')
    pass

if __name__ == '__main__':
    main()
