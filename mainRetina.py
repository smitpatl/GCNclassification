import tensorflow as tf
import random
import numpy as np
from utils import *
from graph import Graph
import scipy.sparse as sp
from layers import GraphConvLayer

class GCN():
    def __init__(self,G,layer_sizes,has_features,learning_rate=0.01,epochs=200,activation=tf.nn.relu,train_ratio=0.1):
        self.train_ratio = train_ratio
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.layer_sizes = layer_sizes
        self.activation = activation
        self.g = G
        self.adj,self.labels,self.features,self.train_mask,self.val_mask,self.test_mask = preprocess_data(G,train_ratio,has_features)
        self.build_placeholders()

    def build_placeholders(self):
        num_supports = 1
        self.placeholders = {
            'adj': tf.sparse_placeholder(tf.float32),
            'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(self.features[2], dtype=tf.int64)),
            'labels': tf.placeholder(tf.float32, shape=(None, self.labels.shape[1])),
            'labels_mask': tf.placeholder(tf.int32),
            'is_training': tf.placeholder(tf.bool)
        }
'''***************************************************************************************
*   Model code inspired from
*	https://github.com/tkipf/gcn
***************************************************************************************'''
    def gcn(self):
        L = len(self.layer_sizes)
        input_size = self.features[2][1]
        y = self.placeholders['features']
        for i in range(L):
            if i==0:
                sparse = True
            else:
                sparse = False
                y = tf.layers.dropout(y,0.5,training = self.placeholders['is_training'])
            y = GraphConvLayer(input_dim=input_size, output_dim=self.layer_sizes[i],name='gc%d'%i,
                              activation=self.activation)(adj_norm=self.placeholders['adj'],
                                            x= y, sparse=sparse)
            input_size = self.layer_sizes[i]
        loss = self.cross_entropy(y,self.placeholders['labels'],self.placeholders['labels_mask'])
        print(self.placeholders['labels_mask'])
        accuracy = self.masked_accuracy(y,self.placeholders['labels'],self.placeholders['labels_mask'])
        opt = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)
        return y,loss,accuracy,opt


    def construct_feed_dict(self, labels_mask, is_training):
        feed_dict = dict()
        feed_dict.update({self.placeholders['is_training']: is_training})
        feed_dict.update({self.placeholders['labels']: self.labels})
        feed_dict.update({self.placeholders['labels_mask']: labels_mask})
        feed_dict.update({self.placeholders['features']: self.features})
        feed_dict.update({self.placeholders['adj']: self.adj})
        return feed_dict


    def cross_entropy(self, preds, labels, mask):
        loss = tf.losses.log_loss(labels, preds)
        mask = tf.cast(mask, dtype=tf.float32)
        mask /= tf.reduce_mean(mask)
        loss *= mask
        return tf.reduce_mean(loss)


    def masked_accuracy(self, preds, labels, mask):
        correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
        accuracy_all = tf.cast(correct_prediction, tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)
        mask /= tf.reduce_mean(mask)
        accuracy_all *= mask
        return tf.reduce_mean(accuracy_all)

    def train_and_evaluate(self):
        output,loss,accuracy,opt = self.gcn()
        config = tf.ConfigProto()
        sess = tf.InteractiveSession(config=config)
        init = tf.global_variables_initializer()
        sess.run(init)

        feed_dict_train = self.construct_feed_dict(self.train_mask,True)
        feed_dict_val = self.construct_feed_dict(self.val_mask,False)
        feed_dict_test = self.construct_feed_dict(self.test_mask,False)

        for i in range(self.epochs):
            loss_tr,_,acc_tr = sess.run([loss,opt,accuracy],feed_dict = feed_dict_train)
            if (i+1)%10 == 0:
                loss_v,acc_v = sess.run([loss,accuracy],feed_dict = feed_dict_val)
                print('-'*150)
                print('step {:d} \t train_loss = {:.3f} \t train_accuracy =  {:.3f} \t val_loss = {:.3f} \t val_accuracy = {:.3f}'.format(i+1,loss_tr,acc_tr,loss_v,acc_v))

        acc_te = sess.run(accuracy,feed_dict = feed_dict_test)
        print('-'*150)
        print('after training, test accuracy is %f'%acc_te)

def main():
    layer_sizes = [16,8,2]
    train_ratio=0.05
    learning_rate=0.001
    epochs=800
    has_features = False
    activation = tf.nn.softmax
    weighted=False
    directed=False
    G = Graph('data/GroundTruth.edges',weighted,directed,'data/GroundTruth.labels',None)
    model = GCN(G,layer_sizes,has_features,learning_rate,epochs,activation,train_ratio)
    model.train_and_evaluate()

if __name__ == '__main__':
    main()
