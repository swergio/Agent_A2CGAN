import numpy as np
import tensorflow as tf

from Model.Utility.ops import sample
from Model.Utility.layers import fc, batch_normalization

from Model.Generator.lstm import lstm


def Generator(fh, env, nbatch,embeddingScope, reuse = False, init_X_msg = None, max_msg_length = None, initLenght = None):
   
    action_space = env.action_space
    ncty = action_space[0]
    ncnr = action_space[1]
    nmty = action_space[2]
    nmsg = action_space[3]
    nact = 1
    nmax = max(ncty,ncnr,nmty,nmsg,nact)

    fh_cty = fh
    fh_cnr = fh
    fh_mty = fh
    fh_msg = fh
    fh_act = fh  
    msg_length = env.LengthOfMessageText
    
    with tf.variable_scope("generator", reuse=reuse):
               
        cty = fc(fh_cty, 'cty', ncty, act=lambda x:x)
        cnr = fc(fh_cnr, 'cnr', ncnr, act=lambda x:x)
        mty = fc(fh_mty, 'mty', nmty, act=lambda x:x)
        mesh = fc(fh,'mesh', nmsg)
        mesc = np.zeros(mesh.shape)
        mess = tf.concat(axis=1, values=[mesc, mesh])
    
        if init_X_msg is None:
            inital_x = env.MessageText.PadIndex(env.MessageText.IndexSartMessage(nbatch), env.message_space[3])
        else:
            inital_x = init_X_msg
            
        mes, state, mask,txt = lstm(inital_x, mess,msg_length, "meslstm",embeddingScope,mesh.shape[1].value,env,steps  = max_msg_length,initLenght= initLenght )
        act = fc(fh,'act',1,act=lambda x:x)

        cty0 = tf.reshape(sample(cty),[nbatch,1])
        cnr0 = tf.reshape(sample(cnr),[nbatch,1])
        mty0 = tf.reshape(sample(mty),[nbatch,1])
        mes0 = txt
        act0 = tf.reshape(tf.sigmoid(act),[nbatch,1])
        act0 =  tf.cast(tf.greater_equal(act0,tf.zeros(act0.shape)*0) ,tf.int64)
        a0 = tf.concat([cty0,cnr0,mty0,mes0,act0], axis = 1)
      
        a_logits = (cty,cnr,mty,mes,act)
    return a0,a_logits,mask





