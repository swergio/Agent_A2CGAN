import tensorflow as tf

def message_encoding(X, env, name = "message_encoding",XasOneHot = False):
        message_space = env.message_space

        if XasOneHot is False:
            X_cty, X_cnr, X_mty, X_msg, X_act = tf.split(X,num_or_size_splits = message_space, axis = 1)

        if XasOneHot:
            X_cty =X[0]
            X_cnr = X[1]
            X_mty = X[2]
            X_msg = X[3]
            X_act = X[4]
            X_msg = tf.reshape(X_msg,[X_msg.shape[0].value,X_msg.shape[1].value*X_msg.shape[2].value])
        else:
            vocab_space = env.vocab_space
            ncty = vocab_space[0]
            ncnr = vocab_space[1]
            nmty = vocab_space[2]
            nmsg = vocab_space[3]

            X_cty = env.ChatType.IndexToOneHot(X_cty)
            X_cnr = env.ChatNumber.IndexToOneHot(X_cnr)
            X_mty = env.MessageType.IndexToOneHot(X_mty)
            X_msg = env.MessageText.IndexToOneHot(X_msg)
            X_act = tf.to_float(X_act)

            X_cty = tf.reshape(X_cty,[X_cty.shape[0].value,X_cty.shape[1].value*X_cty.shape[2].value])
            X_cnr = tf.reshape(X_cnr,[X_cnr.shape[0].value,X_cnr.shape[1].value*X_cnr.shape[2].value])    
            X_mty = tf.reshape(X_mty,[X_mty.shape[0].value,X_mty.shape[1].value*X_mty.shape[2].value])
            X_msg = tf.reshape(X_msg,[X_msg.shape[0].value,X_msg.shape[1].value*X_msg.shape[2].value])
            #X_act = tf.reshape(X_act,[X_act.shape[0],X_act.shape[1]*X_act.shape[2]])
        
        X_f = tf.concat([X_cty,X_cnr,X_mty,X_msg,X_act],axis = 1)

        return X_f