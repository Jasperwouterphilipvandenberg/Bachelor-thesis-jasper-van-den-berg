import matplotlib.pyplot as plt
import numpy as np
import matplot2tikz

def sym(x):
    #the ratio of amount of elements on and above the diagonal relative to the total amount of elements in square matrix
    if x<=0:
        return 0
    return (x+1)/(2*x)

def flops_mult(n1,n2,r,m):
    #the amount of operations it takes for matrix multiplication
    if n1<=0 or n2<=0 or r<=0:
        return 0
    return n1*n2*(r*(m+1)-1)

def flops_sym_mult(n1,r,m):
    #the amount of operations it takes for matrix multiplication
    if n1<=0 or r<=0:
        return 0
    return n1**2*(r*(m+1)-1)*sym(n1)

def not_zero(x):
    #returns 1 if x is not zero and 0 if it is
    if x==0:
        return 0
    return 1

def both_not_zero(x,y):
    #returns 1 if both x and y are not zero and 0 if they both are
    if x==0 or y==0:
        return 0
    return 1

def strassen(d,n1,r1,n2,r2,y1,x1,x2,y2,m):
    #returns the operations needed for the pebble game Strassen algorithm where the first matrix has n1 x r1 dimension and the second has r2 x n2 dimension
    #the first matrix has non-zero dimension y1 x x1 and the second y2 x x2
    #if a non-zero dimension is zero, zero operations needed
    if x1==0 or y1==0 or x2==0 or y2==0:
        return 0
    #if no more iterations can be made, return reular multiplication
    if x1==1 or y1==1 or x2==1 or y2==1:
        return flops_mult(y1,x2,min(x1,y2),m)
    else:
        #check if dimensions are even and pad if needed
        if n1%2==0:
            n1_half = n1/2
        else:
            n1_half = (n1+1)/2
        if r1%2==0:
            r1_half = r1/2
        else:
            r1_half = (r1+1)/2
        if n2%2==0:
            n2_half = n2/2
        else:
            n2_half = (n2+1)/2
        if r2%2==0:
            r2_half = r2/2
        else:
            r2_half = (r2+1)/2 
    #compute the non-zero dimensions for the submatrices
    n1_r,n2_r,r1_r,r2_r = min(y1,n1_half),min(x2,r1_half),min(x1,n2_half),min(y2,r2_half)
    n1_0,n2_0,r1_0,r2_0 = max(y1-n1_r,0),max(x2-n2_r,0),max(x1-r1_r,0),max(y2-r2_r,0)          
    
    #the operations needed for additions within the Strassen algorithm
    const = (3*n1_0*r1_0+n1_r*r1_0+n1_0*r1_r+3*n2_0*r2_0+n2_r*r2_0+n2_0*r2_r+
    (n1_r*n2_r+2*n1_r*n2_0+2*n1_0*n2_r+n1_r*n2_r)*both_not_zero(r1_0,r2_0)+(2*n1_0*n2_0+2*n1_r*n2_0)*not_zero(r1_0)*(1-not_zero(r2_0))+(n1_0*n2_0+3*n1_0*n2_r)*not_zero(r2_0)*(1-not_zero(r1_0))+(1-not_zero(r1_0))*(1-not_zero(r2_0))*(2*n1_0*n2_0))
    #set the minimum depth for the cutoff criteria to be included
    if d>=0:
        #the cutoff criteria
        if flops_mult(y1,x2,min(y2,x1),m)<const+flops_mult(n1_r,n2_r,min(r1_r,r2_r),m) + flops_mult(n1_r,n2_r,min(r1_0,r2_0),m) + flops_mult(n1_0,n2_r,min(r1_r,r2_r),m) + flops_mult(n1_r,n2_0,min(r1_0,r2_r),m) + flops_mult(n1_0,n2_r,min(r1_r,r2_0),m) + flops_mult(n1_r,n2_0,min(r1_r,r2_r),m)+ flops_mult(n1_0,n2_0,min(r1_r,r2_r),m)+ flops_mult(n1_0,n2_r-n2_0,min(r1_r,r2_0),m)+ flops_mult(n1_r-n1_0,n2_0,min(r1_0,r2_r),m)+ flops_mult(n1_r-n1_0,n2_r-n2_0,min(r1_0,r2_0),m):            
            return flops_mult(y1,x2,min(y2,x1),m)
    #returning the operation cost of the additions and recursive calls of the cost equations
    return const + strassen(d+1,n1_half,r1_half,n2_half,r2_half,n1_r,r1_r,n2_r,r2_r,m) + strassen(d+1,n1_half,r1_half,n2_half,r2_half,n1_r,r1_0,n2_r,r2_0,m) + strassen(d+1,n1_half,r1_half,n2_half,r2_half,n1_0,r1_r,n2_r,r2_r,m) + strassen(d+1,n1_half,r1_half,n2_half,r2_half,n1_r,r1_0,n2_0,r2_r,m) + strassen(d+1,n1_half,r1_half,n2_half,r2_half,n1_0,r1_r,n2_r,r2_0,m)+ strassen(d+1,n1_half,r1_half,n2_half,r2_half,n1_r,r1_r,n2_0,r2_r,m) + strassen(d+1,n1_half,r1_half,n2_half,r2_half,n1_0,r1_r,n2_0,r2_r,m)+ strassen(d+1,n1_half,r1_half,n2_half,r2_half,n1_0,r1_r,n2_r-n2_0,r2_0,m)+ strassen(d+1,n1_half,r1_half,n2_half,r2_half,n1_r-n1_0,r1_0,n2_0,r2_r,m)+ strassen(d+1,n1_half,r1_half,n2_half,r2_half,n1_r-n1_0,r1_0,n2_r-n2_0,r2_0,m)

def strassen_sym(d,n,r,y1,x1,y2,m):
    #returns the operations needed for the pebble game Strassen algorithm with symmetric result where the first matrix has n x r dimension and the second has r x n dimension
    #the first matrix has non-zero dimension y1 x x1 and the second y2 x y1
    #if a non-zero dimension is zero, zero operations needed
    if r==0 or x1==0:
        return 0
    if x1==1 or y1==1:
        return flops_sym_mult(x1,y1,m)
    else:
        if n%2==0:
            n_half = n/2
        else:
            n_half = (n+1)/2
        if r%2==0:
            r_half = r/2
        else:
            r_half = (r+1)/2
        n1_r,r1_r,r2_r = min(y1,n_half),min(x1,r_half),min(y2,r_half)
        n1_0,r1_0,r2_0 = max(y1-n1_r,0),max(x1-r1_r,0),max(y2-r2_r,0)          
        
        const = (4*n1_0*r1_0+3*n1_0*r2_0+n1_r*r2_0+n1_0*r2_r+
        ((2*n1_r*n1_r*sym(n1_r)+2*n1_r*n1_0*sym(n1_0))+n1_0*n1_r)*both_not_zero(r1_0,r2_0)+(2*n1_0*n1_0+n1_r*n1_0*sym(n1_0))*not_zero(r1_0)*(1-not_zero(r2_0))+((2+sym(n1_0))*n1_0*n1_r)*not_zero(r2_0)*(1-not_zero(r1_0))+(1-not_zero(r1_0))*(1-not_zero(r2_0))*(n1_0*n1_0))
        if d>=0:
            if flops_sym_mult(y1,min(y2,x1),m)<const+flops_sym_mult(n1_r,min(r1_r,r2_r),m) + flops_sym_mult(n1_r,min(r1_0,r2_0),m) + flops_mult(n1_0,n1_r,min(r1_r,r2_r),m) + flops_sym_mult(n1_0,min(r1_0,r2_r),m) + flops_mult(n1_0,n1_r,min(r1_r,r2_0),m) + flops_sym_mult(n1_0,min(r1_r,r2_r),m)+ flops_mult(n1_r-n1_0,n1_0,min(r1_r,r2_r),m)+ flops_mult(n1_0,n1_r-n1_0,min(r1_r,r2_0),m)+ flops_sym_mult(n1_r-n1_0,min(r1_0,r2_0),m):
                
                return flops_sym_mult(y1,min(y2,x1),m)
        return const + strassen_sym(d+1,n_half,r_half,n1_r,r1_r,r2_r,m) + strassen_sym(d+1,n_half,r_half,n1_r,r1_0,r2_0,m) + strassen(d+1,n_half,r_half,n_half,r_half,n1_0,r1_r,n1_r,r2_r,m) + strassen_sym(d+1,n_half,r_half,n1_0,r1_0,r2_r,m) + strassen(d+1,n_half,r_half,n_half,r_half,n1_0,r1_r,n1_r,r2_0,m) + strassen_sym(d+1,n_half,r_half,n1_0,r1_r,r2_r,m)+ strassen(d+1,n_half,r_half,n_half,r_half,n1_r-n1_0,r1_0,n1_0,r2_r,m)+ strassen_sym(d+1,n_half,r_half,n1_r-n1_0,r1_0,r2_0,m)

def strassen_w(d,n1,r1,n2,r2,y1,x1,x2,y2,m):
    #returns the operations needed for the Winograd Strassen algorithm where the first matrix has n1 x r1 dimension and the second has r2 x n2 dimension
    #the first matrix has non-zero dimension y1 x x1 and the second y2 x x2
    #if a non-zero dimension is zero, zero operations needed

    if x1==0 or y1==0 or x2==0 or y2==0:
        return 0
    #if no more iterations can be made, return reular multiplication
    if x1==1 or y1==1 or x2==1 or y2==1:
        return flops_mult(y1,x2,min(x1,y2),m)
    else:
        #check if dimensions are even and pad if needed
        if n1%2==0:
            n1_half = n1/2
        else:
            n1_half = (n1+1)/2
        if r1%2==0:
            r1_half = r1/2
        else:
            r1_half = (r1+1)/2
        if n2%2==0:
            n2_half = n2/2
        else:
            n2_half = (n2+1)/2
        if r2%2==0:
            r2_half = r2/2
        else:
            r2_half = (r2+1)/2    
        
        #compute the non-zero dimensions for the submatrices
        n1_r,n2_r,r1_r,r2_r = min(y1,n1_half),min(x2,n2_half),min(x1,r1_half),min(y2,r2_half)
        n1_0,n2_0,r1_0,r2_0 = max(y1-n1_r,0),max(x2-n2_r,0),max(x1-r1_r,0),max(y2-r2_r,0)    
        #the operations needed for additions within the Strassen algorithm
        const = (n1_0*(r1_0+2*r1_r)+r1_0*n1_r+ n2_0*(r2_r+2*r2_0)+r2_0*n2_r
                 + n1_r*n2_r*(both_not_zero(r1_0,r2_0)+1)+n1_0*n2_r*(2+not_zero(r1_0))+n1_r*n2_0*(not_zero(r2_0)+1))
        #set the minimum depth for the cutoff criteria to be included
        if d>=0:
            #the cutoff criteria
            if flops_mult(y1,x2,min(x1,y2),m)<const+2*flops_mult(n1_r,n2_r,min(r1_r,r2_r),m) + flops_mult(n1_r,n2_r,min(r1_0,r2_0),m) + flops_mult(n1_0,n2_r,min(r1_r,r2_r),m) + flops_mult(n1_r,n2_0,min(r1_r,r2_r),m) + flops_mult(n1_r,n2_0,min(r1_r,r2_0),m) + flops_mult(n1_0,n2_r,min(r1_0,r2_r),m):
                return flops_mult(y1,x2,min(x1,y2),m)
        #returning the operation cost of the additions and recursive calls of the cost equations
        return const + 2*strassen_w(d+1,n1_half,r1_half,n2_half,r2_half,n1_r,r1_r,n2_r,r2_r,m) + strassen_w(d+1,n1_half,r1_half,n2_half,r2_half,n1_r,r1_0,n2_r,r2_0,m) + strassen_w(d+1,n1_half,r1_half,n2_half,r2_half,n1_0,r1_r,n2_r,r2_r,m) + strassen_w(d+1,n1_half,r1_half,n2_half,r2_half,n1_r,r1_r,n2_0,r2_r,m) + strassen_w(d+1,n1_half,r1_half,n2_half,r2_half,n1_r,r1_r,n2_0,r2_0,m) + strassen_w(d+1,n1_half,r1_half,n2_half,r2_half,n1_0,r1_0,n2_r,r2_r,m)
        
def strassen_w_sym(d,n,r,y1,x1,y2,m):
    #returns the operations needed for the Winograd Strassen algorithm with symmetric result where the first matrix has n x r dimension and the second has r x n dimension
    #the first matrix has non-zero dimension y1 x x1 and the second y2 x y1
    #if a non-zero dimension is zero, zero operations needed
    if r==0 or x1==0:
        return 0
    #if no more iterations can be made, return reular multiplication
    if x1==1 or y1==1:
        return flops_sym_mult(y1,min(x1,y2),m)
    
    else:
        #check if dimensions are even and pad if needed
        if n%2==0:
            n_half = n/2
        else:
            n_half = (n+1)/2
        if r%2==0:
            r_half = r/2
        else:
            r_half = (r+1)/2
        #compute the non-zero dimensions for the submatrices
        n1_r,r1_r,r2_r = min(y1,n_half),min(x1,r_half),min(y2,r_half)
        n1_0,r1_0,r2_0 = max(y1-n1_r,0),max(x1-r1_r,0), max(y2-r2_r,0)    
        #the operations needed for additions within the Strassen algorithm
        const = (n1_0*(r1_0+r2_0*2+r2_r+2*r1_r)+n1_r*r2_0
                 + n1_r**2*(sym(n1_r)*both_not_zero(r1_0,r2_0)+1)+(2+not_zero(r1_0))*n1_r*n1_0)
        #set the minimum depth for the cutoff criteria to be included
        if d>=0:
            #the cutoff criteria
            if flops_sym_mult(y1,min(x1,y2),m)<const+flops_sym_mult(n1_r,r1_r,m) + flops_sym_mult(n1_r,r1_0,m) + flops_sym_mult(n1_0,r1_r,m) +flops_mult(n1_r,n1_r,r1_r,m)+ flops_mult(n1_r,n1_0,r1_r,m) + flops_mult(n1_0,n1_r,r1_0,m):          
                return flops_sym_mult(y1,min(x1,y2),m)
        #returning the operation cost of the additions and recursive calls of the cost equations
        return const + strassen_w_sym(d+1,n_half,r_half,n1_r,r1_r,r2_r,m) +strassen_w(d+1,n_half,r_half,n_half,r_half,n1_r,r1_r,n1_r,r2_r,m) +strassen_w_sym(d+1,n_half,r_half,n1_r,r1_0,r2_0,m) + strassen_w_sym(d+1,n_half,r_half,n1_0,r1_r,r2_r,m) + strassen_w(d+1,n_half,r_half,n_half,r_half,n1_r,r1_r,n1_0,r1_r,m) + strassen_w(d+1,n_half,r_half,n_half,r_half,n1_0,r1_0,n1_r,r1_0,m)


def standard_cholesky(n,m):
    #operation cost eqaution of the standard algorithm of the Cholesky factorization
    return (m+1)/6*n**3+n**2*(m/2)+n*(m/3-2/12)


def blocked_cholesky_dyna_w(d,n,r,m):
    #The operation cost equation of the blocked Cholesky factorization with matrix multiplication using Winograd's strassen algorithm with static padding
    #if the matrix has no elements, no factorization can be made
    if n<=0:
        return 0
    #If no more iterations can be made, regular Cholesky factorization algorithm can be used
    if n-r<=0:
        return standard_cholesky(n,m)
    else:
        const = (n-r)**2+(n-r)*(r*(r+1)/2*m+r*(r-1)/2)+strassen_w_sym(0,n-r,r,n-r,r,r,m)
        #set the minimum depth for the cutoff criteria to be included
        if d>=0:
            #the cutoff criteria
            if 1*standard_cholesky(n,m)-standard_cholesky(n-r,m)-standard_cholesky(r,m)<=const:             
                
                return standard_cholesky(n,m)
        #returning the operation cost of the steps and recursive calls of the cost equations
        return standard_cholesky(r,m)+const +blocked_cholesky_dyna_w(d+1,n-r,r,m)

def blocked_cholesky_dyna(d,n,r,m):
    #The operation cost equation of the blocked Cholesky factorization with matrix multiplication using Winograd's strassen algorithm with static padding
    #if the matrix has no elements, no factorization can be made
    if n<=0:
        return 0
    #If no more iterations can be made, regular Cholesky factorization algorithm can be used
    if n-r<=0:
        return standard_cholesky(n,m)
    else:
        const = (n-r)**2+(n-r)*(r*(r+1)/2*m+r*(r-1)/2)+strassen_sym(0,n-r,r,n-r,r,r,m)
        #set the minimum depth for the cutoff criteria to be included
        if d>=0:
            #the cutoff criteria
            if standard_cholesky(n,m)-standard_cholesky(n-r,m)-standard_cholesky(r,m)<=const:                     
                
                return standard_cholesky(n,m)
        #returning the operation cost of the steps and recursive calls of the cost equations
        return standard_cholesky(r,m)+const +blocked_cholesky_dyna(d+1,n-r,r,m)
    
    
        
def main_cholesky(n_max,steps,r,d):
    #the function that plots the operation cost equation of the blocked cholesky equation with various strassen algorithms
    #n_max is the maximum matrix size tested, steps is how many points will be calculated, r is the blocked cholesky step-size, d is the digit count (2^d)
    
    #list of multiplication constants
    list_mult_ratios = [1, 2, 4, 8, 16, 32, 55.0, 64.0, 73.0, 82.0, 91.0, 100.0, 109.0, 118.0, 127.0, 136.0, 145.0, 154.0, 163.0, 172.0]
    decimals_power = 10
    m = list_mult_ratios[decimals_power-1]
    #In order to not compute and plot the operation cost equation of a variant, four lines corresponding to said variant have to be made comments
    standard_list = []
    cho_d_w =[]
    cho_d =[]
    #repeat 
    if n_max%steps!=0:
        raise ValueError("max size not divisible by steps")
    x = n_max/steps
    for i in range(steps):
        print(i)
        #add the result of operation cost equation to list
        standard_list.append(standard_cholesky(x*i,m))
        cho_d_w.append(blocked_cholesky_dyna_w(0,x*i,r,m))
        cho_d.append(blocked_cholesky_dyna(0,x*i,r,m))
        
        
    index = np.linspace(1,n_max,steps)
    standard_list = np.array(standard_list)
    plt.figure(figsize=(8, 6))
    #plot the operation cost equation
    plt.plot(index,standard_list,"r-", label = "Standard Cholesky")
    plt.plot(index,cho_d_w,"b-", label = "Dynamic Winograd")
    plt.plot(index,cho_d,"g-", label = "Dynamic Pebble game")

    
    #print difference between the standard algorithm and the blocked Cholesky factorization, negative procent means a speed-up
    print("rate:",(-standard_list[steps-1]+cho_d_w[steps-1])/standard_list[steps-1]*100)
    print("rate:",(-standard_list[steps-1]+cho_d[steps-1])/standard_list[steps-1]*100)
    
    plt.grid(True)
    plt.legend()
    plt.xlabel("Matrix size")
    plt.ylabel("Operation count")
    plt.show()
    matplot2tikz.save("plot.tex")

def main_strassen(n_max,steps,r,d):
    #the function that plots the operation cost equation of the Strassen algorithm
    #n_max is the maximum matrix size tested, steps is how many points will be calculated, r is the blocked cholesky step-size, d is the digit count (2^d)
    
    #list of multiplication constants
    list_mult_ratios = [1, 2, 4, 8, 16, 32, 55.0, 64.0, 73.0, 82.0, 91.0, 100.0, 109.0, 118.0, 127.0, 136.0, 145.0, 154.0, 163.0, 172.0]
    decimals_power = 10
    m = list_mult_ratios[decimals_power-1]

    #In order to not compute and plot the operation cost equation of a variant, four lines corresponding to said variant have to be made comments
    standard_list = []
    cho_d_w =[]
    cho_d =[]
    #repeat 
    if n_max%steps!=0:
        raise ValueError("max size not divisible by steps")
    x = n_max/steps
    for i in range(steps):
        print(i)
        #add the result of operation cost equation to list
        standard_list.append(flops_sym_mult(x*i,r,m))
        cho_d_w.append(strassen_w_sym(0,x*i,r,x*i,r,r,m))
        cho_d.append(strassen_sym(0,x*i,r,x*i,r,r,m))
        
        
    index = np.linspace(1,n_max,steps)
    standard_list = np.array(standard_list)
    plt.figure(figsize=(8, 6))
    #plot the operation cost equation
    plt.plot(index,standard_list,"r-", label = "Standard Cholesky")
    plt.plot(index,cho_d_w,"b-", label = "Dynamic Winograd")
    plt.plot(index,cho_d,"g-", label = "Dynamic Pebble game")
    
    #print difference between the standard algorithm and the blocked Cholesky factorization, negative procent means a speed-up
    print("rate:",(-standard_list[steps-1]+cho_d_w[steps-1])/standard_list[steps-1]*100)
    print("rate:",(-standard_list[steps-1]+cho_d[steps-1])/standard_list[steps-1]*100)

    
    plt.grid(True)
    plt.legend()
    plt.xlabel("Matrix size")
    plt.ylabel("Operation count")
    plt.show()
    matplot2tikz.save("plot.tex")