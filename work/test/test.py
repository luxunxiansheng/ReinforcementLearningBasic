import tensorflow as tf  
indices = [[3], [5], [0], [7]]  
print(indices)
indices = tf.reshape(indices, (1, 4))
print(indices)  
a = tf.one_hot(indices, depth=10, on_value=None, off_value=None, axis=None, dtype=None, name=None)  
print ("a is : ")  
print (a)  
b = tf.reshape(a, (4, 10))  
print ("a is : ")  
print(b) 
