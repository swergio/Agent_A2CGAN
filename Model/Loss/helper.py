import tensorflow as tf
from Model.Utility.ops import cat_entropy 

def softmax_adjusted_loss(logits,labels, ADV, M,ent_coef ):
    neglogpac = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    pg_loss = tf.reduce_sum(ADV * neglogpac*M)/tf.reduce_sum(M)
    entropy= tf.reduce_sum(cat_entropy(logits)*M)/tf.reduce_sum(M)
    loss = (pg_loss - entropy*ent_coef)
    return loss

def softmax_adjusted_loss_2dim(logits,labels,mask, ADV, M,ent_coef ):
    
    neglogpac_1 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    mask = tf.to_float(mask)
    neglogpac = tf.reduce_sum(neglogpac_1 *  mask, [1]) #/tf.reduce_sum(mask,[1])
    pg_loss = tf.reduce_sum(ADV * neglogpac*M)/tf.reduce_sum(M)
    cat_ent = cat_entropy(logits, dim = 2)
    entropy_1 = tf.reduce_sum(cat_ent * mask,[1]) #/tf.reduce_sum(mask,[1])
    entropy= tf.reduce_sum(entropy_1*M)/tf.reduce_sum(M)
    loss = (pg_loss - entropy*ent_coef)
    return loss

def sigmoid_adjusted_loss(logits,labels, ADV, M,ent_coef ):
    neglogpac = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
    pg_loss = tf.reduce_sum(ADV * neglogpac*M)/tf.reduce_sum(M)
    entropy= tf.reduce_sum(cat_entropy(logits)*M)/tf.reduce_sum(M)
    loss = (pg_loss - entropy*ent_coef)
    return loss

def GAN_generator_feedback(Dfake):
    D_fake = Dfake.d
    G_loss = tf.clip_by_value(D_fake,0,1) - 0.5 
    return G_loss