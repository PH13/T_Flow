

import numpy as np

import tensorflow as tf


matrix1 = [[1.0, 2.0], [3.0, 40]]
matrix2 = np.array([[1.0, 2.0], [3.0, 40]], dtype=np.float32)
matrix3 = tf.constant([[1.0, 2.0], [3.0, 40]])

print(type(matrix1))
print(type(matrix2))
print(type(matrix3))


tensorForM1 = tf.convert_to_tensor(matrix1, dtype=tf.float32)
tensorForM2 = tf.convert_to_tensor(matrix2, dtype=tf.float32)
tensorForM3 = tf.convert_to_tensor(matrix3, dtype=tf.float32)

print(type(tensorForM1))
print(type(tensorForM2))
print(type(tensorForM3))


mat1 = tf.constant([[4, 5, 6],[3,2,1]])
mat2 = tf.constant([[7,8,9],[10,11,12]])

# hadamard product (element wise)
mult = tf.multiply(mat1, mat2)

# dot product (no. of rows = no. of columns)
dotprod = tf.matmul(mat1, tf.transpose(mat2))

with tf.Session() as sess:
    print(sess.run(mult))
    print(sess.run(dotprod))

mat = tf.constant([
    [0, 1, 2],
    [3, 4, 5],
    [6, 7, 8]
], dtype = tf.float32)

# get trace ('sum of diagonal elements') of the matrix
mat = tf.trace(mat)

with tf.Session() as sess:
    print(sess.run(mat))


