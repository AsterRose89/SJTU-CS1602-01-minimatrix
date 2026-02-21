# Test code for IEEE course final project
# Fan Cheng, 2024

import minimatrix as mm

"""
测试1↓↓↓
"""

mat = mm.Matrix([[1, 2, 3], [6, 5, 4], [7, 8, 9]])

print(mat)  # 顺便验证了__str__函数
print(mat.data)
print(mat.dim)
print(mat.shape())  # 返回矩阵的各项参数

# print(mm.Matrix(None, None))  # 无有效输入，会报错，报错信息为：请输入参数！
print(mm.Matrix(None, (2, 3)))  # 输出全零矩阵
print(mm.Matrix([[1, 2]], None))

m1 = mm.Matrix([[1, 2]], (2, 3))
print(m1.dim)  # 输出：(1,2)，自动修正Matrix的dim

print(mat.reshape((1, 9)))
print(mat)  # 原mat不变
# print(mat.reshape((2, 2)))  # 若格式不匹配，会报错，报错信息为：矩阵维数不匹配！

A = mm.Matrix(data=[[1, 2], [3, 4]])
print(A.dot(A))

mat.T()
print(mat)  # 转置一次
mat.T()
print(mat)  # 转置两次，得到原来的mat

A = mm.Matrix(data=[[1, 2, 3], [4, 5, 6]])
print(A.sum(0))
print(A.sum(1))
print(A.sum())

mat_copy = mat.copy()
print(mat_copy)
mat_copy.T()
print(mat_copy)
print(mat)  # mat本身不因为其copy()的转置而转置

print(mat.Kronecker_product(mat))

print(mat[1, 2])
print(mat[1:2, 0:2])
print(mat[:2, :])

x = mm.Matrix(data=[
    [0, 1, 2, 3],
    [4, 5, 6, 7],
    [8, 9, 0, 1]
])
x[1, 2] = 0
print(x)
x[1:, 2:] = mm.Matrix(data=[[1, 2], [3, 4]])
print(x)

print(mat ** 0)
print(mat ** 2)

m1 = mm.Matrix([[1, 2, 3], [6, 5, 4], [7, 8, 9]])
m2 = mm.Matrix([[1, 1, 2], [2, 2, 3], [4, 0, 7]])
print(m1 + m2)
print(m1 - m2)
print(m1 * m2)

print(len(mat))

print(mat.gauss())
print(mat.det())
print(mat.inverse())  # 行列式为0，会报错，报错信息为：输入矩阵行列式为零，逆矩阵不存在！
print(mat.rank())

m3 = mm.Matrix([[1, 2, 3], [3, 5, 2], [6, 8, 3]])
print(m3.gauss())
print(m3.det())
print(m3.inverse())
print(m3.rank())

print(mm.Matrix.I(3))
print(mm.Matrix.I(5))

print(mm.Matrix.narray((2, 3, 4), 2))
print(mm.Matrix.narray((1, 5)))

print(mm.Matrix.arange(1, 10, 2))
print(mm.Matrix.arange(10, 2, -1))

print(mm.Matrix.zeros((2, 3, 4)))
print(mm.Matrix.zeros_like(mm.Matrix(data=[[1, 2, 3], [2, 3, 4]])))
print(mm.Matrix.ones((2, 3, 4)))
print(mm.Matrix.ones_like(mm.Matrix(data=[[1, 2, 3], [2, 3, 4]])))

print(mm.Matrix.nrandom((3, 2)))
print(mm.Matrix.nrandom((2, 3, 4)))

print(mm.Matrix.nrandom_like(mm.Matrix(data=[[1, 2, 3], [2, 3, 4]])))

mat = mm.Matrix([[1, 2, 3], [6, 5, 4], [7, 8, 9]])
mat1 = mm.Matrix([[0, 0, 3], [1, 1, 4], [2, 3, 9]])
print(mm.Matrix.concatenate((mat, mat1), 0))
print(mm.Matrix.concatenate((mat, mat1), 1))


def f(x):
    return x ** 2


def g(x):
    return -x


print(mat.vectorize(f))
print(mat.vectorize(g))


"""
测试2↓↓↓
"""

m24 = mm.Matrix.arange(0, 24, 1)
print(m24)
print(m24.reshape([3, 8]))
print(m24.reshape([24, 1]))
print(m24.reshape([4, 6]))

print(mm.Matrix.zeros((3, 3)))
print(mm.Matrix.zeros_like(m24))

print(mm.Matrix.ones((3, 3)))
print(mm.Matrix.ones_like(m24))

print(mm.Matrix.nrandom((3, 3)))
print(mm.Matrix.nrandom_like(m24))


# 最小二乘法
def least_square_method(m, n):
    X = mm.Matrix.nrandom((m, n))
    w = mm.Matrix.nrandom((n, 1))
    lst = mm.Matrix.nrandom((m, 1)).data
    ave = sum(i[0] for i in lst) / len(lst)
    for j in lst:
        j[0] -= ave
    e = mm.Matrix(lst)
    Y = X.dot(w) + e

    X_T = X.copy()
    X_T.T()
    ww = X_T.dot(X).inverse().dot(X_T).dot(Y)

    return w - ww


print(least_square_method(1000, 100))