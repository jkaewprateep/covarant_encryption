# https://stackoverflow.com/questions/71867730/tensorflow-vector-times-vector-multiplication
# https://www.tensorflow.org/api_docs/python/tf/tensordot
# https://www.tensorflow.org/api_docs/python/tf/einsum
# https://www.tensorflow.org/api_docs/python/tf/linalg/matmul

import tensorflow as tf

import matplotlib.pyplot as plt


dataset = { "Type_A": 33070.953, "Type_B": 35263.8, "Type_C": 34265.55, "Type_D": 30398.55, "Type_E": 31593.15, "Type_F": 32591.4, "Type_G": 32059.8, "Type_H": 31303.55, }
mapping = { "Type_A": 0.33, "Type_B": 0.33, "Type_C": 0.33, "Type_D": 0.33, "Type_E": 0.33, "Type_F": 0.33, "Type_G": 0.25, "Type_H": 0.25,}


Cell_A = 0.33
Cell_B = 0.25
Cell_C = 0.33

volatilities = tf.constant([[ Cell_A, Cell_B, Cell_C ]], dtype=tf.float32)
correlations = tf.constant([[ 33405, 33405, 33405 ], [ 33405, 40050, 33405 ], [ 33405, 33405, 37025 ]], dtype=tf.float32)

volatility_matrix = tf.einsum('ij,jk->ik', volatilities, correlations)
correlation_matrix = correlations

print( ' volatilities : ' )
print( volatilities )
print( ' correlations : ' )
print( correlations )
print( ' volatility_matrix : ' )
print( volatility_matrix )

print( ' Shortest: ' + str( tf.math.argmin( volatility_matrix[0] ).numpy() ) )
print( ' Hghest: ' + str( tf.math.argmax( volatility_matrix[0] ).numpy() ) )

image_1 = tf.constant([ Cell_A, Cell_B, Cell_C], shape=(1, 3), dtype=tf.float32)
image_2 = tf.constant(correlations, shape=(3, 3), dtype=tf.float32)
image_3 = tf.constant(volatility_matrix, shape=(1, 3), dtype=tf.float32)

plt.figure(figsize=(4, 1))
plt.suptitle("Matrixes co-varants")

plt.subplot(4, 1, 1)
plt.xticks([])
plt.yticks([])
plt.grid(False)
plt.imshow( image_1 )
plt.xlabel( "Original" )

plt.subplot(4, 1, 2)
plt.xticks([])
plt.yticks([])
plt.grid(False)
plt.imshow( image_2 )
plt.xlabel( "Matrixes co-relations" )

plt.subplot(4, 1, 3)
plt.xticks([])
plt.yticks([])
plt.grid(False)
plt.imshow( image_3 )
plt.xlabel( "Volatility Matrix" )


volatility_matrix = tf.einsum('ij,jk->ik', volatilities, correlations)

item_name_1 = [ mapping[x] for ( x, y ) in dataset.items() if y == volatility_matrix[0][0]]
item_name_2 = [ mapping[x] for ( x, y ) in dataset.items() if y == volatility_matrix[0][1]]
item_name_3 = [ mapping[x] for ( x, y ) in dataset.items() if y == volatility_matrix[0][2]]

image_4 = tf.constant([ item_name_1, item_name_2, item_name_3 ], shape=(1, 3))

plt.subplot(4, 1, 4)
plt.xticks([])
plt.yticks([])
plt.grid(False)
plt.imshow( image_4 )
plt.xlabel( "Mapping method" )

plt.show()
