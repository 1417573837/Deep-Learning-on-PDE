# -*- coding:utf-8 -*-
"""
Author:Alarak
Date:2023/07/31

Vector Dense Layer
This name is from initial inspiration.
General speaking, I used 2D activations here to add the express ability to the neural network.
May extend to nD activations.
How it works:
receiving m -dimension input vector and then separate it into nD subvectors, with amount m/n.
Then apply all l activations on each subvector, to get m/n*l -dimension outputs.
"""
import tensorflow as tf
act1 = lambda x,y:tf.sin(x+y)
act2 = lambda x,y:tf.sin(x*y)
act3 = lambda x, y: tf.sin(x) * tf.sin(y)

class VectorDenseLayer(tf.keras.layers.Layer):
    def __init__(self, num_inputs, activations,**kwargs):
        super(VectorDenseLayer, self).__init__()
        self.num_inputs = num_inputs
        self.activations = [act1,act2,act3] #a list of activation funcs


    def call(self, inputs):  # input shape=[batch_size, num_inputs]
        assert self.num_inputs == int(inputs.shape[1]), "num_input error"
        # self.batch_size = int(inputs.shape[0])

        num_acts=len(self.activations)
        act_dim=self.activations[0].__code__.co_argcount

        assert self.num_inputs % act_dim == 0, "act_dim Mismatch"
        x=[]
        y=[]
        for i in range(act_dim):
            x.append(inputs[:,int(i*self.num_inputs/act_dim):(i+1)*int(self.num_inputs/act_dim)])
        for j in range(num_acts):
            # print(x)
            y.append(self.activations[j](*x))
        outputs=tf.concat(y, axis=1)
        return outputs

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_inputs": self.num_inputs,
            "activations": self.activations
        })
        return config
