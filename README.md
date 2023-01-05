# covarant_encryption
For study covarants encryptions 

```
Cell_A = 0.33
Cell_B = 0.25
Cell_C = 0.33

volatilities = tf.constant([[ Cell_A, Cell_B, Cell_C ]], dtype=tf.float32)
correlations = tf.constant([[ 33405, 33405, 33405 ], [ 33405, 40050, 33405 ], [ 33405, 33405, 37025 ]], dtype=tf.float32)

volatility_matrix = tf.einsum('ij,jk->ik', volatilities, correlations)
```

```
volatility_matrix = tf.einsum('ij,jk->ik', volatilities, correlations)

item_name_1 = [ mapping[x] for ( x, y ) in dataset.items() if y == volatility_matrix[0][0]]
item_name_2 = [ mapping[x] for ( x, y ) in dataset.items() if y == volatility_matrix[0][1]]
item_name_3 = [ mapping[x] for ( x, y ) in dataset.items() if y == volatility_matrix[0][2]]

image = tf.constant([ item_name_1, item_name_2, item_name_3 ], shape=(1, 3))
```

## Results ##

#### Signals and reverse mapping techniques ( local table ) ####

![Alt text](https://github.com/jkaewprateep/covarant_encryption/blob/main/Figure_2.png "Title")

#### Signals and reverse mapping techniques ( local table ) ####

![Alt text](https://github.com/jkaewprateep/covarant_encryption/blob/main/Figure_3.png "Title")


#### Signals and reverse mapping techniques ( local table ) ####

![Alt text](https://github.com/jkaewprateep/covarant_encryption/blob/main/Figure_4.png "Title")


#### Signals and reverse mapping techniques ( local table ) ####

![Alt text](https://github.com/jkaewprateep/covarant_encryption/blob/main/Figure_5.png "Title")
