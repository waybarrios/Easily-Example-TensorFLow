#Import useful libs
"""
@Author: Waybarrios
Based on Tensorfow Docs

"""
import tensorflow as tf
import numpy as np
import input_data as data



"""
MNIST URL: http://yann.lecun.com/exdb/mnist/

This tutorial is intended for readers who are new to both machine learning and TensorFlow. 
you already know what MNIST is, and what softmax (multinomial logistic) regression is, you might prefer this faster paced tutorial.

When one learns how to program, there's a tradition that the first thing you do is print "Hello World." 
Just like programming has Hello World, machine learning has MNIST.

"""
#Downloading MNIST dataset

mnist = data.read_data_sets("MNIST_data/", one_hot=True)


"""
x isn't a specific value. It's a placeholder, 
a value that we'll input when we ask TensorFlow to run a computation. 

We want to be able to input any number of MNIST images, 
each flattened into a 784-dimensional vector.
 We represent this as a 2d tensor of floating point numbers, 
with a shape [None, 784]. (Here None means that a dimension can be of any length.)

"""

x = tf.placeholder("float", [None, 784])

"""
We also need the weights and biases for our model. 
We could imagine treating these like additional inputs, 
but TensorFlow has an even better way to handle it: Variable. 

A Variable is a modifiable tensor that lives in TensorFlow's graph of interacting operations. 
It can be used and even modified by the computation. 
For machine learning applications, one generally has the model parameters be Variables.

"""

W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

#Sotfmax model 

"""

First, we multiply x by W with the expression tf.matmul(x,W). 
This is flipped from when we multiplied them in our equation, 
where we had Wx, as a small trick to deal with x being a 2D tensor with multiple inputs. 
We then add b, and finally apply tf.nn.softmax.

"""
y = tf.nn.softmax(tf.matmul(x,W) + b)

"""
To implement cross-entropy we need to 
first add a new placeholder to input the correct answers.
"""

y_ = tf.placeholder("float", [None,10]) #Y_PRIME (TRUE DISTRIBUTIONS LABELS WITH CORRECT ANSWERS)

#Implement cross-entropy function

"""
First, tf.log computes the logarithm of each element of y. 
Next, we multiply each element of y_ with the corresponding element of tf.log(y_)

Finally, tf.reduce_sum adds all the elements of the tensor.

"""

cross_entropy = - tf.reduce_sum(y_*tf.log(y))

"""

TensorFlow know the entire graph of your computations, 
it can automatically use the backpropagation algorithm 
efficiently determine how your variables affect the cost you ask it minimize

"""

#it can apply your choice of optimization algorithm to modify the variables and reduce the cost.

"""
We ask TensorFlow to minimize cross_entropy using the gradient descent algorithm with
a learning rate of 0.01. 

Gradient descent is a simple procedure, where TensorFlow simply shifts each 
a little bit in the direction that reduces the cost
"""

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)


# Now we have our model set up to train.

"""
before we launch it, we have to add an operation to initialize the variables we created
"""

init = tf.initialize_all_variables()

#We can now launch the model in a Session, and run the operation that initializes the variables

sess = tf.Session()
sess.run(init)

"""
---------------------------------------------------------------------------------------

GO AHEAD!!!

It's time to train

---------------------------------------------------------------------------------------

"""

#Training step 1000 times!

print ":::: Training mode is comming :::: "


for i in range (1000):
	print " Traning Step %d with batch 100" % i
	batch_xs, batch_ys = mnist.train.next_batch(100) #xs means image array | ys label array
	sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})


"""
Each step of the loop, we get a "batch" of one hundred random data points from our training set. 
We run train_step feeding in the batches data to replace the placeholders.

Using small batches of random data is called stochastic training -- in this case, stochastic gradient descent. 
Ideally, we'd like to use all our data for every step of training because that would give us a better 
sense of what we should be doing, but that's expensive. 
So, instead, we use a different subset every time. 
Doing this is cheap and has much of the same benefit.

"""

"""
---------------------------------------------------------------------------------------

Evaluating our model

---------------------------------------------------------------------------------------

"""

print ":::: Evaluating mode is comming :::: "

"""

Well, first let's figure out where we predicted the correct label. tf.argmax is an extremely useful function 
which gives you the index of the highest entry in a tensor along some axis. 
For example, tf.argmax(y,1) is the label our model thinks is most likely for each input,
while tf.argmax(y_,1) is the correct label. We can use tf.equal to check if our prediction matches the truth.

"""

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))


"""

That gives us a list of booleans. 
To determine what fraction are correct, we cast to floating point numbers and then take the mean. 

For example, [True, False, True, True] would become [1,0,1,1] which would become 0.75.

"""

accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

#Show accuracy
print "Accuracy is %f" % sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
