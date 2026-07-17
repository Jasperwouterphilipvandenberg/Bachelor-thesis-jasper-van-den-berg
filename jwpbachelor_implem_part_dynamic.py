from flint import*
import random
import time
import itertools
import numpy as np
import numba as nb
import math as mt
import matplotlib.pyplot as plt
from matplotlib import scale as mscale
from matplotlib.transforms import Transform, IdentityTransform
from matplotlib.scale import ScaleBase
import tikzplotlib

def genRandom(n, m):
    #Calculates a n by m matrix with random numbers between 1 and 2, which is divided by another random number to ensure the right precision (works up to 1024 digits at least)
    res = np.empty((n, m))
    for i in nb.prange(n):
        for j in range(m):
            res[i, j] = (np.random.rand()+1)
    return arb_mat(res.tolist())/arb(np.random.rand())

def add_columns_rows(A,x,y):
    #Add x zero columns to A
    if x==0 and y==0:
        return A
    L = A.table()
    for i in range(x):
        for j in range(A.nrows()):
            L[j].append(0)
    for i in range(y):
        L.append([0]*(x+A.ncols()))
    return arb_mat(L)

def del_cols_rows(A,x,y):
    #Delete x columns from A
    if x==0 and y==0:
        return A
    L = arb_mat([[0]*(A.ncols()-x)]*(A.nrows()-y))
    for i in range(A.ncols()-x):
        for j in range(A.nrows()-y):
            L[j,i] = A[j,i]
    return L

def mult(A,B,m,x1,y1,x2,y2):
    if A.ncols()-x1<=0 or B.nrows()-y2<=0:
        return arb_mat([[0]*(A.nrows())]*(B.ncols()))
    
    L = arb_mat([[0]*(B.ncols())]*(A.nrows()))

    for i in range(A.nrows()-y1):
        for j in range(B.ncols()-x2):
            L[i,j] = sum(A[i,k]*B[k,j] for k in range(min(A.ncols()-x1,B.nrows()-y2)))

    return L

def mult_sym(A,B,m,x1,y1,x2,y2):
    if A.ncols()-x1<=0 or B.nrows()-y2<=0:
        return arb_mat([[0]*(A.nrows())]*(B.ncols()))
    L = arb_mat([[0]*(B.ncols())]*(A.nrows()))
    for i in range(A.nrows()-y1):
        for j in range(i):
            L[i,j]=L[j,i] = sum(A[i,k]*B[k,j] for k in range(min(A.ncols()-x1,B.nrows()-y2)))
        L[i,i] = sum(A[i,k]*B[k,i] for k in range(min(A.ncols()-x1,B.nrows()-y2)))

    return L

def add_sym(A,B,y1,x2):

    if A.ncols()-y1<=0 or B.nrows()-x2<=0:
        return arb_mat([[0]*(A.nrows())]*(B.ncols()))
    L = arb_mat([[0]*(B.ncols())]*(A.nrows()))

    for i in range(A.nrows()-y1):
        for j in range(i):
            L[i,j]=L[j,i] = A[i,j]+B[i,j]
        L[i,i] = A[i,i]+B[i,i]
    return L

def split_matrix(A):
    if A.ncols()%2 !=0 and A.nrows()%2!=0:
        raise ValueError("dimensions not divisible by two")
    A11,A12,A21,A22=arb_mat([[0]*int(A.ncols()/2)]*int(A.nrows()/2)),arb_mat([[0]*int(A.ncols()/2)]*int(A.nrows()/2)),arb_mat([[0]*int(A.ncols()/2)]*int(A.nrows()/2)),arb_mat([[0]*int(A.ncols()/2)]*int(A.nrows()/2))
    for i in range(int(A.nrows()/2)):
        for j in range(int(A.ncols()/2)):
            A11[i,j] = A[i,j]
            A12[i,j] = A[i,j+A.ncols()/2]
            A21[i,j] = A[i+A.nrows()/2,j]
            A22[i,j] = A[i+A.nrows()/2,j+A.ncols()/2]
    return A11,A12,A21,A22

def combine_matrix(A11,A12,A21,A22):
    if A11.ncols()!=A12.ncols()!=A22.ncols() or A11.nrows()!=A12.nrows()!=A12.nrows():
        raise ValueError("Dimensions do not match")
    A = arb_mat([[0]*(2*A12.ncols())]*(2*A12.nrows()))

    for i in range(A11.nrows()):
        for j in range(A11.ncols()):
            A[i,j] = A11[i,j]
            A[i,j+A11.ncols()] = A12[i,j]
            A[i+A11.nrows(),j] = A21[i,j]
            A[i+A11.nrows(),j+A11.ncols()] = A22[i,j]
    return A

def sym(x):
    #Return the number of elements in a matrix on and below the matrix divided by the total number of elements of the matrix
    if x==0:
        raise ValueError("nulletje")
    return (x+1)/(2*x)

def flops_mult(n1,n2,r,m):
    #Return the operation cost of standard matrix multiplication
    if n1==0 or n2==0 or r==0:
        return 0
    return n1*n2*(r*(m+1)-1)

def flops_sym_mult(n1,r,m):
    #Return the operation cost of standard matrix multiplication where the result is symmetric
    if n1==0 or r==0:
        return 0
    return n1**2*(r*(m+1)-1)*sym(n1)

def dynamic_w(A,B,m,x1,y1,x2,y2):
    #Winograd Strassen algorithm, m is the multiplication constant, x1,y1,x2,y2 represent the zero rows and columns of A and B respectively.
    a,b,c,d=0,0,0,0
    #Check if more iterations can be made or if matrices are non-zero, if they are, use standard multiplication 
    if A.ncols()-x1<=1 or A.nrows()-y1<=1 or B.ncols()-x2<=1 or B.nrows()-y2<=1:
        if (x1<=1 and y1<=1) and (x2<=1 and y2<=1):
            res = A*B
        else:
            res = mult(A,B,m,x1,y1,x2,y2)
            
    else: 
        
        
        #Add rows or columns if dimension are odd
        if A.ncols()%2!=0:
            a=1
            x1 += 1
        if A.nrows()%2!=0:
            b=1
            y1 += 1
        A = add_columns_rows(A,a,b)
        if B.ncols()%2!=0:
            c=1
            x2 += 1
        if B.nrows()%2!=0:
            d=1
            y2 += 1
        A = add_columns_rows(A,c,d)
        #Set up new non zero dimensions
        x1_r,y1_r,x2_r,y2_r = min(x1,A.ncols()/2),min(y1,A.nrows()/2),min(x2,B.ncols()/2),min(y2,B.nrows()/2)
        x1_0,y1_0,x2_0,y2_0 = max(0,x1-A.ncols()/2),max(0,y1-A.nrows()/2),max(0,x2-B.ncols()/2),max(0,y2-B.nrows()/2)
        #Cutoff criteria, addition may be included in the steps, but isn't yet here
        if flops_mult(A.nrows(),B.ncols(),A.ncols(),m)<=(2*flops_mult(A.nrows()-y1_0, B.ncols()-x2_0, min(A.ncols()-x1_0,B.nrows()-y2_0),m)+flops_mult(A.nrows()-y1_0, B.ncols()-x2_0, min(A.ncols()-x1_r,B.nrows()-y2_r),m)+flops_mult(A.nrows()-y1_r, B.ncols()-x2_0, min(A.ncols()-x1_0,B.nrows()-y2_0),m)+flops_mult(A.nrows()-y1_0, B.ncols()-x2_r, min(A.ncols()-x1_0,B.nrows()-y2_0),m)+flops_mult(A.nrows()-y1_0, B.ncols()-x2_r, min(A.ncols()-x1_0,B.nrows()-y2_r),m)+flops_mult(A.nrows()-y1_r, B.ncols()-x2_0, min(A.ncols()-x1_r,B.nrows()-y2_0),m)
        +4*A.nrows()/4*A.ncols()+4*B.ncols()/4*A.ncols()+7*B.ncols()*A.nrows()/4):
           
            if (x1<=1 and y1<=1) and (x2<=1 and y2<=1):
                res = A*B
            else:
                res = mult(A,B,m,x1,y1,x2,y2)
        else:
            #Steps of Strassen algorithm, where non-zero dimensions are kept track of properly through iterations
            A11,A12,A21,A22 = split_matrix(A)
            B11,B12,B21,B22 = split_matrix(B)
            S1,S3,T1,T3 = A21+A22,A11-A21,B12-B11,B22-B12
            S2,T2 = S1-A11,B22-T1
            S4,T4 = A12-S2,B21-T2
            
            P1,P2,P3,P4,P5,P6,P7 = dynamic_w(A11,B11,m,x1_0,y1_0,x2_0,y2_0),dynamic_w(A12,B21,m,x1_r,y1_0,x2_0,y2_r),dynamic_w(S1,T1,m,x1_0,y1_r,x2_0,y2_0),dynamic_w(S2,T2,m,x1_0,y1_0,x2_0,y2_0),dynamic_w(S3,T3,m,x1_0,y1_0,x2_r,y2_0),dynamic_w(S4,B22,m,x1_0,y1_0,x2_r,y2_r),dynamic_w(A22,T4,m,x1_r,y1_r,x2_0,y2_0) 
            U1,U2 = P1+P2,P1+P4
            U3,U6 = U2+P5,U2+P3
            U4,U5,U7 = U3+P7,U3+P3,U6+P6
            res = combine_matrix(U1,U7,U4,U5)
    #Delete the rows added by padding
    
    return del_cols_rows(res,b,a)
    

def dynamic_sym_w(A,B,m,x1,y1,x2,y2):
    #Winograd Strassen algorithm with symmetric result, m is the multiplication constant, x1,y1,x2,y2 represent the zero rows and columns of A and B respectively.
    a,b,c,d=0,0,0,0
    #Check if more iterations can be made or if matrices are non-zero, if they are, use standard multiplication 
    if A.ncols()-x1<=1 or A.nrows()-y1<=1 or B.ncols()-x2<=1 or B.nrows()-y2<=1:
        if (x1<=1 and y1<=1) and (x2<=1 and y2<=1):
            res = A*B
        else:
            res = mult_sym(A,B,m,x1,y1,x2,y2)
    else:  
        
        #Add rows or columns if dimension are odd
        if A.ncols()%2!=0:
            a=1
            x1 += 1
        if A.nrows()%2!=0:
            b=1
            y1 += 1
        A = add_columns_rows(A,a,b)
        if B.ncols()%2!=0:
            c=1
            x2 += 1
        if B.nrows()%2!=0:
            d=1
            y2 += 1
        A = add_columns_rows(A,c,d)
        #Set up new non zero dimensions
        x1_r,y1_r,x2_r,y2_r = min(x1,A.ncols()/2),min(y1,A.nrows()/2),min(x2,B.ncols()/2),min(y2,B.nrows()/2)
        x1_0,y1_0,x2_0,y2_0 = max(0,x1-A.ncols()/2),max(0,y1-A.nrows()/2),max(0,x2-B.ncols()/2),max(0,y2-B.nrows()/2)
        #Cutoff criteria, addition may be included in the steps, but isn't yet here
        if flops_sym_mult(A.nrows(),A.ncols(),m)<=(flops_sym_mult(A.nrows()-y1_0, min(A.ncols()-x1_0,B.nrows()-y2_0),m)+flops_sym_mult(A.nrows()-y1_0, min(A.ncols()-x1_r,B.nrows()-y2_r),m)+flops_sym_mult(A.nrows()-y1_r, min(A.ncols()-x1_0,B.nrows()-y2_0),m)+flops_mult(A.nrows()-y1_0, B.ncols()-x2_0, min(A.ncols()-x1_0,B.nrows()-y2_0),m)+flops_mult(A.nrows()-y1_0, B.ncols()-x2_r, min(A.ncols()-x1_0,B.nrows()-y2_0),m)+flops_mult(A.nrows()-y1_r, B.ncols()-x2_0, min(A.ncols()-x1_r,B.nrows()-y2_0),m)):         
            if (x1<=1 and y1<=1) and (x2<=1 and y2<=1):
                res = A*B
            else:
                res = mult_sym(A,B,m,x1,y1,x2,y2)
        else:
            #Steps of Strassen algorithm, where non-zero dimensions are kept track of properly through iterations
            A11,A12,A21,A22 = split_matrix(A)
            B11,B12,B21,B22 = split_matrix(B)
            S1,S3,T1,T3 = A21+A22,A11-A21,B12-B11,B22-B12
            S2,T2 = S1-A11,B22-T1
            T4 = B21-T2
            P1,P2,P3,P4,P5,P7 = mult_sym(A11,B11,m,x1_0,y1_0,x2_0,y2_0),mult_sym(A12,B21,m,x1_r,y1_0,x2_0,y2_r),mult_sym(S1,T1,m,x1_0,y1_r,x2_0,y2_0),mult (S2,T2,m,x1_0,y1_0,x2_0,y2_0),mult(S3,T3,m,x1_0,y1_0,x2_r,y2_0),mult(A22,T4,m,x1_r,y1_r,x2_0,y2_0) 
            U1,U2 = P1+P2,P1+P4
            U3 = U2+P5
            U4,U5 = U3+P7,add_sym(U3,P3,y1_r,x2_0)

            res = combine_matrix(U1,U4.transpose(),U4,U5)
    #Delete the rows added by padding
    return del_cols_rows(res,b,a)
    

def dynamic(A,B,m,x1,y1,x2,y2):
    #Pebble game Strassen algorithm, m is the multiplication constant, x1,y1,x2,y2 represent the zero rows and columns of A and B respectively.
    a,b,c,d=0,0,0,0
    #Check if more iterations can be made or if matrices are non-zero, if they are, use standard multiplication 
    if A.ncols()-x1<=1 or A.nrows()-y1<=1 or B.ncols()-x2<=1 or B.nrows()-y2<=1:
        if (x1<=1 and y1<=1) and (x2<=1 and y2<=1):
            res = A*B
        else:
            res = mult(A,B,m,x1,y1,x2,y2)
    else:  
        
        
        #Add rows or columns if dimension are odd
        if A.ncols()%2!=0:
            a=1
            x1 += 1
        if A.nrows()%2!=0:
            b=1
            y1 += 1
        A = add_columns_rows(A,a,b)
        if B.ncols()%2!=0:
            c=1
            x2 += 1
        if B.nrows()%2!=0:
            d=1
            y2 += 1
        A = add_columns_rows(A,c,d)
        #Set up new non zero dimensions
        x1_r,y1_r,x2_r,y2_r = min(x1,A.ncols()/2),min(y1,A.nrows()/2),min(x2,B.ncols()/2),min(y2,B.nrows()/2)
        x1_0,y1_0,x2_0,y2_0 = max(0,x1-A.ncols()/2),max(0,y1-A.nrows()/2),max(0,x2-B.ncols()/2),max(0,y2-B.nrows()/2)
        #Cutoff criteria, addition may be included in the steps, but isn't yet here
        if flops_mult(A.nrows(),B.ncols(),A.ncols(),m)<=(2*flops_mult(A.nrows()-y1_0, B.ncols()-x2_0, min(A.ncols()-x1_0,B.nrows()-y2_0),m)+flops_mult(A.nrows()-y1_0, B.ncols()-x2_0, min(A.ncols()-x1_r,B.nrows()-y2_r),m)+flops_mult(A.nrows()-y1_r, B.ncols()-x2_0, min(A.ncols()-x1_0,B.nrows()-y2_0),m)+flops_mult(A.nrows()-y1_0, B.ncols()-x2_0, min(A.ncols()-x1_r,B.nrows()-y2_r),m)+flops_mult(A.nrows()-y1_r, B.ncols()-x2_0, min(A.ncols()-x1_0,B.nrows()-y2_r),m)+flops_mult(A.nrows()-y1_0, B.ncols()-x2_r, min(A.ncols()-x1_0,B.nrows()-y2_0),m)):         
            if (x1<=1 and y1<=1) and (x2<=1 and y2<=1):
                res = A*B
            else:
                res = mult(A,B,m,x1,y1,x2,y2)
        else:
            #Steps of Strassen algorithm, where non-zero dimensions are kept track of properly through iterations
            A11,A12,A21,A22 = split_matrix(A)
            B11,B12,B21,B22 = split_matrix(B)
            A22,B22 = A12-A21+A22,B12-B21+B22
            S1,S2,S3,T1,T2,T3 = A21+A22,A22-A12,A22-A11,B22-B11,B21+B22,B22-B12      
            P1,P2,P3,P4,P5,P6,P7 = mult(A11,B11,m,x1_0,y1_0,x2_0,y2_0),mult(A12,B21,m,x1_r,y1_0,x2_0,y2_r),mult(A21,T1,m,x1_0,y1_r,x2_0,y2_0),mult (A22,B22,m,x1_r,y1_0,x2_0,y2_r),mult(S1,T2,m,x1_0,y1_r,x2_0,y2_r),mult(S2,T3,m,x1_0,y1_r,x2_0,y2_r),mult(S3,B12,m,x1_0,y1_0,x2_r,y2_0)
            C11,C12,C21,C22 = P1+P2,P5-P7,P3+P6,P5+P6-P2-P4
            C12 = C12-C22
            C21=C22-C21
            res = combine_matrix(C11,C12,C21,C22)
    #Delete the rows added by padding
    return del_cols_rows(res,b,a)

def dynamic_sym(A,B,m,x1,y1,x2,y2):
    #Pebble game Strassen algorithm with symmetric result, m is the multiplication constant, x1,y1,x2,y2 represent the zero rows and columns of A and B respectively.
    a,b,c,d=0,0,0,0
    #Check if more iterations can be made or if matrices are non-zero, if they are, use standard multiplication 
    if A.ncols()-x1<=1 or A.nrows()-y1<=1 or B.ncols()-x2<=1 or B.nrows()-y2<=1:
        if (x1<=1 and y1<=1) and (x2<=1 and y2<=1):
            res = A*B
        else:
            res = mult_sym(A,B,m,x1,y1,x2,y2)
        
    else: 
        
        #Add rows or columns if dimension are odd
        if A.ncols()%2!=0:
            a=1
            x1 += 1
        if A.nrows()%2!=0:
            b=1
            y1 += 1
        A = add_columns_rows(A,a,b)
        if B.ncols()%2!=0:
            c=1
            x2 += 1
        if B.nrows()%2!=0:
            d=1
            y2 += 1
        A = add_columns_rows(A,c,d)
        #Set up new non zero dimensions
        x1_r,y1_r,x2_r,y2_r = min(x1,A.ncols()/2),min(y1,A.nrows()/2),min(x2,B.ncols()/2),min(y2,B.nrows()/2)
        x1_0,y1_0,x2_0,y2_0 = max(0,x1-A.ncols()/2),max(0,y1-A.nrows()/2),max(0,x2-B.ncols()/2),max(0,y2-B.nrows()/2)
        #Cutoff criteria, addition may be included in the steps, but isn't yet here
        if flops_sym_mult(A.nrows(),A.ncols(),m)<=(2*flops_sym_mult(A.nrows()-y1_0, min(A.ncols()-x1_0,B.nrows()-y2_0),m)+flops_sym_mult(A.nrows()-y1_0, min(A.ncols()-x1_r,B.nrows()-y2_r),m)+flops_mult(A.nrows()-y1_0, B.ncols()-x2_0, min(A.ncols()-x1_r,B.nrows()-y2_r),m)+flops_mult(A.nrows()-y1_r, B.ncols()-x2_0, min(A.ncols()-x1_0,B.nrows()-y2_r),m)):  
            if (x1<=1 and y1<=1) and (x2<=1 and y2<=1):
                res = A*B
            else:
                res = mult_sym(A,B,m,x1,y1,x2,y2)
        else:
        #Steps of Strassen algorithm, where non-zero dimensions are kept track of properly through iterations
          A11,A12,A21,A22 = split_matrix(A)
          B11,B12,B21,B22 = split_matrix(B)
          A22,B22 = A12-A21+A22,B12-B21+B22
          S1,S2,S3,T2,T3 = A21+A22,A22-A12,A22-A11,B21+B22,B22-B12      
          P1,P2,P4,P5,P6,P7 = mult_sym(A11,B11,m,x1_0,y1_0,x2_0,y2_0),mult_sym(A12,B21,m,x1_r,y1_0,x2_0,y2_r),mult (A22,B22,m,x1_0,y1_0,x2_0,y2_0),mult_sym(S1,T2,m,x1_0,y1_r,x2_0,y2_r),mult(S2,T3,m,x1_0,y1_r,x2_0,y2_r),mult(S3,B12,m,x1_0,y1_0,x2_r,y2_0)
          temp = P6-P2-P4
          C11,C12,C22 = P1+P2,P5-P7,add_sym(P5,temp,0,0)
          C12 = C12-(P5+temp)
          res = combine_matrix(C11,C12,C12.transpose(),C22)
    #Delete the rows added by padding
    return del_cols_rows(res,b,a)

def strassen_setup(A,B,m):
    #Sets up static padding for pebble game Strassen algorithm
    n=max(A.ncols(),A.nrows())
    for d in range(n):
        if 2**d>=n:
            power = d
            break
    a1,a2,b1,b2 = A.nrows(),A.ncols(),B.ncols(),B.nrows()
    
    A = add_columns_rows(A,2**power-A.ncols(),2**power-A.nrows())
    B = add_columns_rows(B,2**power-B.ncols(),2**power-B.nrows())
    
    #Remove padded rows and columns
    return del_cols_rows(strass_sym(A,B,m,2**power-a1,2**power-a2,2**power-b2,2**power-b1),2**power-b1,2**power-a1)

def strassen_setup_w(A,B,m):
    #Sets up static padding for Winograd Strassen algorithm 
    n=max(A.ncols(),A.nrows())
    for d in range(n):
        if 2**d>=n:
            power = d
            break
    a1,a2,b1,b2 = A.nrows(),A.ncols(),B.ncols(),B.nrows()
    A = add_columns_rows(A,2**power-A.ncols(),2**power-A.nrows())
    B = add_columns_rows(B,2**power-B.ncols(),2**power-B.nrows())
    #Remove padded rows and columns
    
    return del_cols_rows(strass_sym_w(A,B,m,2**power-a1,2**power-a2,2**power-b2,2**power-b1),2**power-b1,2**power-a1)
    

def make_submatrix(A, cols, cole, rows, rowe):
    #Makes a submatrix of A with rows starting at rows and ending at rowe, the same for the columns with cols and cole
    L = arb_mat([[0]*(cole-cols)]*(rowe-rows))
    for i in range(rowe-rows):
        for j in range(cole-cols):
            L[i,j] = A[rows+i,cols+j]
    return L

def merge_submatrix(R_11,R_12,R_22):
    #Combines submartices in the way that is used in blocked cholesky factorisation, so upper right is a zero matrix and R11 and R22 are  lower triangle matrices.
    L = arb_mat([[0]*(R_11.ncols()+R_12.ncols())]*(R_11.ncols()+R_12.ncols()))
    x = max(R_22.ncols(),R_11.ncols())
    for i in range(x):
        for j in range(x):
            if i<R_11.ncols() and j<R_11.ncols():
                L[i,j] = R_11[i,j]
            if j<R_11.ncols() and i<R_22.ncols():
                L[j,i+R_11.ncols()] = R_12[j,i]
            if j<R_22.ncols() and i<R_22.ncols():
                L[i+R_11.ncols(),j+R_11.ncols()]= R_22[i,j]
    return L

def is_same(A,B):
    #Checks if matrices A and B are the same, only used in verifying test results
    for i in range(A.ncols()):
        for j in range(A.nrows()):
            if not A[j,i]-B[j,i].is_zero():
                return False
    return True

def pos_def(A):
    #Checks if a matrix is positive definite, it does not check for symmetry, only if the diagonal dominates the row
    #Only used in the experiment function where we only need a positive definite function, so false negatives are possible
    for i in range(A.ncols()):
        if A[i,i]<=0 or A[i,i]<= (sum((A[i,j]**2)**0.5 for j in range(A.nrows())))-(A[i,i]**2)**0.5:
            return False
    return True

def standard_cholesky(A):
    #Regular cholesky factorisation algorithm
    L = arb("0")*A
    for j in range(A.ncols()):
        for i in range(j):
            L[j,i] = (A[i,j] - sum(L[i,k]*L[j,k] for k in range(i)))/L[i,i]
        if A[j,j] - sum(L[j,k]**2 for k in range(j))<=0:
            raise ValueError("Not positive definite")
        L[j,j] = (A[j,j] - sum(L[j,k]**2 for k in range(j))).root(2)
    return L.transpose()

def strassen_simple_w(n,r,m):
    #Calculates the approximate steps to preform strassen algorithm for a n by r and a r by n matrix. It acts like n and r are powers of 2, in order to calculate it quickly, with a cutoff criteria.
    if n<0 or r<0:
        return 0
    if r<=1 or n<=1:
        return n**2*r*(m+1)-r**2
    if n**2*(r*(m+1)-1)<=7*(n/4)**2*(r/4*(m+1)-1)+2*n*r+7/4*n**2:
        return n**2*(r*(m+1)-1)
    return 7*strassen_simple_w(n/2,r/2,m)+2*n*r+7/4*n**2

def strassen_simple_w_sym(n,r,m):
    #Calculates the approximate steps to preform strassen algorithm for a n by r and a r by n matrix. It acts like n and r are powers of 2, in order to calculate it quickly, with a cutoff criteria.
    if n<0 or r<0:
        return 0
    if r<=1 or n<=1:
        return n**2*(r*(m+1)-1)*sym(n)
    if n**2*(r*(m+1)-1)*sym(n)<=(3+3*sym(n/2))*(n/4)**2*(r/4*(m+1)-1)+7/4*n*r+5/4*n**2:
        return n**2*(r*(m+1)-1)*sym(n)
    return 3*strassen_simple_w_sym(n/2,r/2,m)+3*strassen_simple_w(n/2,r/2,m)+2*n*r+7/4*n**2

def strassen_simple(n,r,m):
    #Calculates the approximate steps to preform strassen algorithm for a n by r and a r by n matrix. It acts like n and r are powers of 2, in order to calculate it quickly, with a cutoff criteria.
    if r<=1 or n<=1:
        return n**2*r*(m+1)-r**2
    if n**2*(r*(m+1)-1)<=7*(n/4)**2*(r/4*(m+1)-1)+6/4*n*r+2*n**2:
        return n**2*(r*(m+1)-1)
    return 7*strassen_simple(n/2,r/2,m)+10/4*n*r+2*n**2

def strassen_simple_sym(n,r,m):
    #Calculates the approximate steps to preform strassen algorithm for a n by r and a r by n matrix. It acts like n and r are powers of 2, in order to calculate it quickly, with a cutoff criteria.
    if r<=1 or n<=1:
        return n**2*r*(m+1)-r**2
    if n**2*(r*(m+1)-1)*sym(n)<=(2+4*sym(n))*(n/4)**2*(r/4*(m+1)-1)+6/4*n*r+2*n**2:
        return n**2*(r*(m+1)-1)
    return 2*strassen_simple(n/2,r/2,m)+4*strassen_simple_sym(n/2,r/2,m)+9/4*n*r+3/2*n**2

def partition_mult(A,m):
    #Matrix multiplication which partitions the matrix in r by r blocks and uses symmetry to remove about half of the calculations
    listy = []
    n = A.ncols()
    r = A.nrows()
    a = mt.floor(n/r)
    b = n-a*r
    #Add zero columns to make the final r by r block
    
    #add all submatrices to a list, which is used later to multiply
    for i in range(a):
        temp = arb_mat([[0]*r]*r)
        for j in range(r):
            for k in range(r):
                temp[j,k] = A[j,k+r*i]
        listy.append(temp)
    temp = arb_mat([[0]*b]*r)
    
    for i in range(r):
        for j in range(b):

            temp[i,j] = A[i,j+a*r]
    listy.append(temp)
    #make the final matrix 
    res = arb_mat([[0]*A.ncols()]*A.ncols())
    for i in range(a):
        #Add matrices on the diagonal
        C = dynamic_sym(listy[i].transpose(),listy[i],m,0,0,0,0)
        for k in range(r):
            for c in range(r):
                
                res[c+i*r,k+i*r] = C[c,k]

        #Add matrices below the diagonal
        for j in range(i):
            C = dynamic(listy[i].transpose(),listy[j],m,0,0,0,0)
            C_t = C.transpose()
            for k in range(r):
                for c in range(r):
                    res[c+i*r,k+j*r] = C[c,k]
                    res[k+j*r,c+i*r] = C_t[k,c]
    #Calculate the bottow row of submatrices
    for i in range(a+1):
        C = dynamic(listy[a].transpose(),listy[i],m,0,0,0,0)
        C_t = C.transpose()

        for j in range(C.ncols()):
            for k in range(C.nrows()):
                res[k+a*r,j+i*r]=C[k,j]
                res[j+i*r,k+a*r]=C_t[j,k]
    return res

def partition_mult_w(A,m):
    #Matrix multiplication which partitions the matrix in r by r blocks and uses symmetry to remove about half of the calculations
    listy = []
    n = A.ncols()
    r = A.nrows()
    a = mt.floor(n/r)
    b = n-a*r
    
    #add all submatrices to a list, which is used later to multiply
    for i in range(a):
        temp = arb_mat([[0]*r]*r)
        for j in range(r):
            for k in range(r):
                temp[j,k] = A[j,k+r*i]
        listy.append(temp)
    temp = arb_mat([[0]*b]*r)
    
    for i in range(r):
        for j in range(b):

            temp[i,j] = A[i,j+a*r]
    listy.append(temp)
    #make the final matrix 
    res = arb_mat([[0]*A.ncols()]*A.ncols())
    for i in range(a):
        #Add matrices on the diagonal
        C = dynamic_sym_w(listy[i].transpose(),listy[i],m,0,0,0,0)
        for k in range(r):
            for c in range(r):
                
                res[c+i*r,k+i*r] = C[c,k]

        #Add matrices below the diagonal
        for j in range(i):
            C = dynamic_w(listy[i].transpose(),listy[j],m,0,0,0,0)
            C_t = C.transpose()
            for k in range(r):
                for c in range(r):
                    res[c+i*r,k+j*r] = C[c,k]
                    res[k+j*r,c+i*r] = C_t[k,c]
    #Calculate the bottow row of submatrices
    for i in range(a+1):
        C = dynamic_w(listy[a].transpose(),listy[i],m,0,0,0,0)
        C_t = C.transpose()

        for j in range(C.ncols()):
            for k in range(C.nrows()):
                res[k+a*r,j+i*r]=C[k,j]
                res[j+i*r,k+a*r]=C_t[j,k]
    return res



def blocked_cholesky_w_part(A,r,m):
    #blocked cholesky algorithm for matrix A with stepsize r  
    n =A.nrows()
    a = mt.ceil((n-r)/r)
    if n-r<=0:
        return standard_cholesky(A)
    #Cutoff criteria, where both regular strassen or partitioned strassen can be used with strassen_simple calculated the approximate steps necessary for matrix multiplication
    if (m+1)/6*(n**3-r**3-(n-r)**3)+(m+1/2)*(n**2-r**2-(n-r)**2)<=(r+1)/(2*(r))*a*(a+1)/2*strassen_simple_w_sym(r,r,m):
        return standard_cholesky(A)
    #If more steps of the algorithm can be made, preform the algorithm
    if n>r:
        A_11 = make_submatrix(A,0,r,0,r)
        A_12 = make_submatrix(A, r, n, 0, r)
        A_22 = make_submatrix(A,r,n,r,n)
        #regular cholesky algorithm
        R_11 = standard_cholesky(A_11)
        #solving the system R_11^T*R_12 = A_12
        R_12 = arb(0)*A_12
        for i in range(n-r):
            for j in range(r):
                R_12[j,i] = (A_12[j,i]-sum(R_11.transpose()[j,k]*R_12[k,i] for k in range(j)))/R_11.transpose()[j,j]
       
        #sum_symm uses symmetry to calculate the summ, partitioned multiplication can be used instead of regular FLINT multiplication
        
        S = A_22- partition_mult_w(R_12,m)
        #Combine the matrices
        L = merge_submatrix(R_11,R_12,blocked_cholesky_w_static(S,r,m))
    #If no more iterations can be preformed
    else:
        L = standard_cholesky(A)
    return L

def blocked_cholesky_part(A,r,m):
    #blocked cholesky algorithm for matrix A with stepsize r  
    n =A.nrows()
    a = mt.ceil((n-r)/r)
    if n-r<=0:
        return standard_cholesky(A)
    #Cutoff criteria, where both regular strassen or partitioned strassen can be used with strassen_simple calculated the approximate steps necessary for matrix multiplication
    if (m+1)/6*(n**3-r**3-(n-r)**3)+(m+1/2)*(n**2-r**2-(n-r)**2)<=(r+1)/(2*(r))*a*(a+1)/2*strassen_simple(r,r,m):
        return standard_cholesky(A)
    #If more steps of the algorithm can be made, preform the algorithm
    
    if n>r:
        A_11 = make_submatrix(A,0,r,0,r)
        A_12 = make_submatrix(A, r, n, 0, r)
        A_22 = make_submatrix(A,r,n,r,n)
        #regular cholesky algorithm
        R_11 = standard_cholesky(A_11)
        #solving the system R_11^T*R_12 = A_12
        R_12 = arb(0)*A_12
        for i in range(n-r):
            for j in range(r):
                R_12[j,i] = (A_12[j,i]-sum(R_11.transpose()[j,k]*R_12[k,i] for k in range(j)))/R_11.transpose()[j,j]
       
        #sum_symm uses symmetry to calculate the summ, partitioned multiplication can be used instead of regular FLINT multiplication
        
        S = A_22- partition_mult(R_12,m)
        #Combine the matrices
        L = merge_submatrix(R_11,R_12,blocked_cholesky_static(S,r,m))
    #If no more iterations can be preformed
    else:
        L = standard_cholesky(A)
    return L


def experiment2(n_max,r,m):
    #The function that runs the experiment for matrices up to n_max by n_max with stepsize r and multiplication cost m, which is based on the precision
    #initialize the resulting lists
    listy_std = []
    listy_blk_s_w = []
    listy_blk_s = []
    listy_blk_d_w = []
    listy_blk_d = []
    listy_blk_p_w = []
    listy_blk_p = []
    listy_blk_p_w_s = []
    listy_blk_p_s = []
    #Iterate for different matrix sizes
    for i in range(n_max,2*n_max):
        temp_standard = []

        temp_blocked_p_w = []
        temp_blocked_p = []

        
        for j in range(2):
            #Ensure that we have a positive definite matrix
            #We take steps of 5 for the matrix dimensions to decrease run time
            matrix = genRandom(1,20*i)
            matrix = matrix.transpose()*matrix
            for _ in range(1000):
                for b in range(i):
                    matrix[b,b] = matrix[b,b]**2 
                if pos_def(matrix):
                    break
            
            #Blocked Cholesky factorization with partitioned Winograd Strassen algorithm
            test = matrix*0
            start = time.time()
            test = blocked_cholesky_w_part(matrix,r,m)
            temp_blocked_p_w.append(time.time()-start)
            if not is_same(test.transpose()*test,matrix):
                raise ValueError("klopt niet")
            test = matrix*0
            #Blocked Cholesky factorization with partitioned pebble game Strassen algorithm
            start = time.time()
            test = blocked_cholesky_part(matrix,r,m)
            temp_blocked_p.append(time.time()-start)
            if not is_same(test.transpose()*test,matrix):
                raise ValueError("klopt niet")
            
            #Regular Cholesky factorization
            test = matrix*0
            start = time.time()
            test = standard_cholesky(matrix)
            temp_standard.append(time.time()-start)
            if not is_same(test.transpose()*test,matrix):
                raise ValueError("klopt niet")
        print(i)    
        #print("voortgang:", sum(temp_blocked_d)/5-sum(temp_standard)/5,sum(temp_blocked_d_w)/5-sum(temp_standard)/5)

        listy_blk_p_w.append(sum(temp_blocked_p_w)/5)
        listy_blk_p.append(sum(temp_blocked_p)/5)     

        listy_std.append(sum(temp_standard)/5)
        
    return listy_std,listy_blk_s_w,listy_blk_s,listy_blk_d_w,listy_blk_d,listy_blk_p_w,listy_blk_p,listy_blk_p_w_s,listy_blk_p_s

def main2(n,r,d):
    #start test, n is the max size of test matrix divided by step size (in this case 5), r is the step size of blocked Cholesky factorization and d is the power-of-two amount of digits (2^d).
    fourier_mult_costs = [1, 2, 4, 8, 16, 32,55, 64,73,82,91,100,109,118,127,136,145]
    ctx.dps = 2**d
    #Set up multiplication constant
    m = fourier_mult_costs[d]
    #Start test
    listy = experiment2(n,r,m)
    index = np.linspace(20*n,40*n,n)
    plt.figure(figsize=(8, 6))
    #Remove the corrseponding plots to exclude certain tests.
    plt.plot(index, listy[0], 'b-', label = "standard Cholesky")
    
    plt.plot(index, listy[5], 'r-', label = "Blocked dynamic Winograd partition")
    plt.plot(index, listy[6], 'r-', label = "Blocked dynamic pebble game partition")
    plt.yscale('cubic')
    plt.grid(True)
    plt.legend()
    plt.xlabel("Matrix size")
    plt.ylabel("Time needed for algorithm(seconds)")
    plt.show()
    x = (listy[5][n-1]-listy[0][n-2])/listy[0][n-1]*100
    y = (listy[6][n-1]-listy[0][n-2])/listy[0][n-1]*100
    tikzplotlib.save("test_cho_p_d_r="+str(r)+"_d=11_w="+str(x)+"s="+str(y)+".tex")




    
