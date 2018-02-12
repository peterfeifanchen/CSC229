#!/usr/bin/python
import re
import numpy as np
import pdb as pdb_
import bdb
import os
import sys
import timeit
import matplotlib.pyplot as plt

def pdb():
    try:
        pdb_.Pdb().set_trace( sys._getframe( 1 ) )
    except bdb.BdbQuit:
        os._exit(1)

def readData( filename, pattern, values ):
    f = open( filename, "r" )
    x_floats = []
    for line in f:
        #print line
        x_values = re.findall(pattern, line)
        for x in x_values:
            values.append(float(x))
            #print float(x) 
    f.close()
    return x_floats

# hessianPartial: returns Hessian(theta_) 
#       = 1/m * sum_m( y^2*x_i*x_j[logisticloss(y_*theta_*x_)(1-logisticloss(y_*theta_*x_)]
# i, j: partial index dH/d_id_j
# x_ : m x n input matrix
# y_ : m x 1 output matrix
# theta_ : n x 1 parameter matrix
def hessianPartial( i, j, x_, y_, theta_ ):
    m = y_.shape[1]
    g = logisticLoss( x_, y_, theta_ )
    imm1 = np.multiply(g, 1-g)
    imm2 = np.multiply(y,y)
    imm3 = np.multiply(x_[i,0:], x_[j,0:])
    res = imm1.dot( np.multiply( imm2, imm3 ).transpose() ) / m
    # res is a 1x1 matrix, we just want the element
    return res.item((0,0))

# gradientPartial: returns partial derivative Jacobian(theta_) 
#       = 1/m * sum_m( y*x_i(1-logisticloss(y_*theta_*x_)) )
# i  : partial derivative dJ/d_i
# x_ : m x n input matrix
# y_ : m x 1 output matrix
# theta_ : n x 1 parameter matrix
def gradientPartial( i, x_, y_, theta_ ):
    m = y_.shape[1]
    g = logisticLoss( x_, y_, theta_ )
    imm1 = np.multiply( y_, x_[i, 0:] )
    res = (1-g).dot( imm1.transpose() )/m
    # res is a 1x1 matrix, we just want the element
    return -res.item((0,0))

# logisticLoss: returns logistic loss function 
#       = 1/(1+e^-(y*theta_*x))
# x_ : m x n input matrix
# y_ : m x 1 output matrix
# theta_ : n x 1 parameter matrix
def logisticLoss( x_, y_, theta_ ):
    t = np.multiply( y_, theta_.dot(x_) )
    return 1/(1 + np.exp(-t))

# logisticRegression: returns prediction for input and parameter
#       = 1/(1+e^-(theta_*x))
# x_ : m x n input matrix
# theta_ : n x 1 parameter matrix
def logisticRegression( x_, theta_ ):
    t = theta_.dot(x_)
    return 1/(1 + np.exp(-t))

# newtonMethod: finds the optimal paramater theta_ given input x_ and output y_ using 
#               newton method theta_ = theta_ - H^-1*J(theta_)
# theta_: a row vector 
# x_: a matrix where each row is a set of inputs and each column is a different parameter
# y_: a vector of outputs corresponding to each row in x_
def newtonMethod( theta_, x_, y_ ):
    # Get number of parameters in theta_ vector
    n = theta_.shape[1]
    # Allocate a Hessian of n x n 
    H = np.matrix( [ [ hessianPartial( i, j, x_, y_, theta_ ) for i in range(0,n) ] \
                     for j in range(0,n) ] )
    # Allocate a Jacobian of n x 1
    J = np.matrix( [ gradientPartial( i, x_, y_, theta_ ) for i in range(0,n) ] )
    return theta_ - (H.I*J.transpose()).transpose()

def verifyResults( x_, y_, theta_, h ):
    m = y_.shape[1]
    p = h( x_, theta_ )
    # All predictions with probability > 0.5 are assigned label {-1}, otherwise label {1}
    p1 = p > 0.5
    r1 = y_ == 1
    diff = np.sum((p1 - r1).astype(int))
    return (m+1-diff)*100/(m+1)

if __name__ == "__main__":
    # read in the floating point values from logistic_x.txt and logistic_y.txt
    file_x = "logistic_x.txt"
    pattern = re.compile("[-]?\d*\.\d+e[\+-]\d+") 
    values = []
    readData( file_x, pattern, values )
    
    x1_floats = values[0:][::2]
    x2_floats = values[1:][::2]
    x3_floats = [1]*len(x1_floats)
    # 99 x 3 matrix of inputs
    x = np.matrix( [x1_floats, x2_floats, x3_floats] )

    file_y = "logistic_y.txt"
    values = []
    pattern = re.compile("[-]?1\.\d+e\+00")
    readData( file_y, pattern, values )

    # 99 x 1 vector of outputs
    y = np.matrix( values[0:] )

    # iterate using Newton's method to find the theta_ that maximus the likelihood 
    # of the observed data
    
    num_iter = 10
    theta_ = np.matrix([ 0, 0, 0 ])
    start = timeit.default_timer()
   
    for i in xrange(10):
        theta_ = newtonMethod( theta_, x, y )

    end = timeit.default_timer()

    # verify accuracy of result
    pc = verifyResults( x, y, theta_, logisticRegression )
    
    print theta_
    print "training time: ", end - start, "s"
    print "accuracy: ", pc, "%"
    

    # matplotlib 1c
    
    x1 = np.array( x[0,0:] ).flatten()
    x2 = np.array( x[1,0:] ).flatten()
    
    lbl = np.empty( (y.shape), dtype=str )
    lbl[y==-1] = '.'
    lbl[y==1] = '+'
    lbl = np.array(lbl).flatten()
    col = np.empty( (y.shape), dtype=str )
    col[y==-1] = 'red'
    col[y==1] = 'blue'
    col = np.array(col).flatten()

    # scatterplot accepts arrays not matrix
    for _lbl, _col, _x1, _x2 in zip( lbl, col, x1, x2 ):
        plt.scatter(_x1, _x2, marker=_lbl, c=_col)
    
    bdry_x1 = np.array( range(0,11) )
    bdry_x2 = (-theta_.item(0,0)*bdry_x1-theta_.item(0,2))/theta_.item(0,1)
    plt.plot(bdry_x1, bdry_x2)

    plt.xlim(0, 10)
    plt.ylim(-5, 5)
    plt.show()
