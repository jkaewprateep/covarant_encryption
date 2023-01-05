# covarant_encryption

For studying covariants encryptions, the mathematicians believe that with summations and covariants, it is impossible to reverse back the original message but it still can by the local table but it will be about 1 - 4 Mbytes for larger mapping values and infinity for precise values.

## Create final value from volatilities and correlations with summations and multiplications ##

In the real world is nearly impossible to know these ratios as you know it is salts or random value when creating an encoder ( see examples in JAVA or DotNet ), transmitting signals is very fast and you never had the final decrypted results than the target do it from assumptions that are why many of computer companies telling you that is not dangerous because it is based on target assumptions and final results never been exposed. 🧸💬 Yes it is but the target possibilities must be large enough or adding random numbers and sequence numbers.

```
Cell_A = 0.33
Cell_B = 0.25
Cell_C = 0.33

volatilities = tf.constant([[ Cell_A, Cell_B, Cell_C ]], dtype=tf.float32)
correlations = tf.constant([[ 33405, 33405, 33405 ], [ 33405, 40050, 33405 ], [ 33405, 33405, 37025 ]], dtype=tf.float32)

volatility_matrix = tf.einsum('ij,jk->ik', volatilities, correlations)
```

## Implementation codes ##

Simple as a Lookup table, game cheaters should stop or the dataset will be very large that is why specific hard drive volumes are used to be intellectual property once that is because they can contain the data dictionary. 😸💬 Not the pocket monsters games 🐑💬 He bought from the bookstores.

```
volatility_matrix = tf.einsum('ij,jk->ik', volatilities, correlations)

item_name_1 = [ mapping[x] for ( x, y ) in dataset.items() if y == volatility_matrix[0][0]]
item_name_2 = [ mapping[x] for ( x, y ) in dataset.items() if y == volatility_matrix[0][1]]
item_name_3 = [ mapping[x] for ( x, y ) in dataset.items() if y == volatility_matrix[0][2]]

image = tf.constant([ item_name_1, item_name_2, item_name_3 ], shape=(1, 3))
```

## Files and Directory ##

| File name | Description |
--- | --- |
| sample.py | sample codes |
| Figure_2.png | Signals and reverse mapping techniques ( local table ) |
| Figure_3.png | Signals and reverse mapping techniques ( local table ) |
| Figure_4.png | Signals and reverse mapping techniques ( local table ) |
| Figure_5.png | Signals and reverse mapping techniques ( local table ) |
| README.md | readme file |

## Results ##

#### Signals and reverse mapping techniques ( local table ) ####

![Alt text](https://github.com/jkaewprateep/covarant_encryption/blob/main/Figure_2.png "Title")

#### Signals and reverse mapping techniques ( local table ) ####

![Alt text](https://github.com/jkaewprateep/covarant_encryption/blob/main/Figure_3.png "Title")


#### Signals and reverse mapping techniques ( local table ) ####

![Alt text](https://github.com/jkaewprateep/covarant_encryption/blob/main/Figure_4.png "Title")


#### Signals and reverse mapping techniques ( local table ) ####

![Alt text](https://github.com/jkaewprateep/covarant_encryption/blob/main/Figure_5.png "Title")
