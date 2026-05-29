import numpy

def f_ex7a_v1(m, n):
    M = numpy.zeros((m, n))
    for i in range(m):
        for j in range(n):
            M[i, j] = i * j
    return M

def f_ex7a_v2(m, n):
    M = numpy.ones((m, n))
    x = numpy.arange(m).reshape((m,1))
    y = numpy.arange(n).reshape((1,n))
    M = M * x
    M = M * y
    return M

def f_ex7a_v3(m, n):
    return numpy.arange(m).reshape((m,1)) * numpy.float64(numpy.arange(n)) # We need to cast one array to float because all computations would be between integeres otherwise, and we would get an integer matrix

def f_ex7b_v1(M):
    M = numpy.array(M)
    arySum = numpy.zeros(M.shape[1])
    for colIdx in range(M.shape[1]):
        for rowIdx in range(M.shape[0]):
            arySum[colIdx] += M[rowIdx, colIdx]
    for rowIdx in range(M.shape[0]):
        for colIdx in range(M.shape[1]):
            M[rowIdx, colIdx] = M[rowIdx, colIdx] / arySum[colIdx]
    return M

def f_ex7b_v2(M):
    arySum = M.sum(0)
    return M / arySum.reshape((1, M.shape[1])) # Reshaping is not needed, but it's useful to put it to remeber we want to treat arySum as a row vector

def f_ex7b_v3(M):
    return M / M.sum(0)

def f_ex7c_v1(M):
    arySum = M.sum(1)
    return M / arySum.reshape((M.shape[0], 1)) # this time resshaping is necessary

def f_ex7c_v2(M):
    return M / M.sum(1).reshape((M.shape[0], 1))

def f_ex7c_v3(M):
    return f_ex7b_v3(M.T).T

def f_ex7d_v1(ary): # We do not know the number of dimensions, so we cannot use for loops directly. We flatten the array, use a for loop and then resshape the resulting array
    aryFlat = ary.flatten() # flattens an array and returns a copy - same as numpy.array(ary.ravel())
    for idx in range(aryFlat.size):
        if aryFlat[idx] < 0:
            aryFlat[idx] = 0
    return aryFlat.reshape(ary.shape) # reshapre the array to the original shape

def f_ex7d_v2(ary): # Use boolean operators and indexing to find and modify the non-zero elements of the array
    ary = numpy.array(ary) # Create a copy
    indexMask = (ary < 0) # Mask of same shape as ary with elements that are True or False based on whether the corresponding ary element is < 0 or not
    ary[indexMask] = 0 # The boolean index operates on the elements for which the corresponding indexMask value is True
    return ary

def f_ex7d_v3(ary): # Exploit numerical operations - False is treated as 0 and True as 1 when combined with non boolean values
    return ary * (ary > 0) # ary > 0 is an array with elements True when the corresponding ary entry is > 0. Multiplying ary by this boolean aray has as effect that elements tof ary that are <= 0 are multipled by False (i.e. by 0), the others are multiplied by True (i.e., by one), thus positive values are unchanged, negative values become 0.
    
def f_ex7e_v1(A, B):
    C = numpy.dot(A, B)
    s = 0
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            s += C[i,j]
    return s

def f_ex7e_v2(A, B):
    C = A @ B # Same as numpy.dot(A, B)
    return C.sum()
    
if __name__ == '__main__':

    print ('a.')
    print (f_ex7a_v1(3, 4))
    print (f_ex7a_v2(3, 4))
    print (f_ex7a_v3(3, 4))

    M = numpy.array([[1.0, 2.0, 6.0, 4.0],
                     [3.0, 4.0, 3.0, 7.0],
                     [1.0, 4.0, 6.0, 9.0]])
    
    print ('\nb.')
    print (f_ex7b_v1(M))
    print (f_ex7b_v2(M))
    print (f_ex7b_v3(M))

    M = numpy.array([[1.0, 3.0, 1.0],
                     [2.0, 4.0, 4.0],
                     [6.0, 3.0, 6.0],
                     [4.0, 7.0, 9.0]]) # Or M = M.T, since this is the transpose of the previous matrix M
    print ('\nc.')
    print (f_ex7c_v1(M))
    print (f_ex7c_v2(M))
    print (f_ex7c_v3(M))

    M = numpy.array([[-1.0, 2.0, 3.0],
                     [2.0, -3.0, -4.0]]) # With 2D arrays
    print ('\nd.')
    print (f_ex7d_v2(M))
    print (f_ex7d_v2(M))
    print (f_ex7d_v2(M))

    x = numpy.array([1.0, -1.0, 2.0, 3.0, -2.0, 4.0, -7.0]) # With 1D arrays
    print (f_ex7d_v2(x))
    print (f_ex7d_v2(x))
    print (f_ex7d_v2(x))
    
    print ('\ne.')
    A = numpy.array([[1.0, 2.0],
                     [3.0, 4.0],
                     [5.0, 6.0]])
    B = numpy.array([[1.0, 3.0],
                     [2.0, 1.0]])
    print (f_ex7e_v1(A, B))
    print (f_ex7e_v2(A, B))
    
