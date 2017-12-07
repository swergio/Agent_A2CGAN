import tensorflow as tf
from Model.Loss.helper import sigmoid_adjusted_loss, softmax_adjusted_loss, softmax_adjusted_loss_2dim
from Model.Utility.ops import mse

def GAN_Discriminator_Loss(Dreal,Dfake):
    D_real = Dreal.d
    D_fake = Dfake.d
    D_loss = 0.5 * (tf.reduce_mean((D_real - 1)**2) + tf.reduce_mean(D_fake**2))
    return D_loss

def GAN_Generator_Loss(Dfake):
    D_fake = Dfake.d
    G_loss = 0.5 * tf.reduce_mean((D_fake - 1)**2)
  
    return G_loss


def Policy_Loss(LogitsModel,A,message_space,ent_coef,M, ADV = None):

    A_cty, A_cnr, A_mty, A_msg, A_act = tf.split(A,num_or_size_splits = message_space, axis = 1)
    A_act = tf.cast(A_act, tf.float32)
    #loss chat type
    cty_coef = 1/5
    loss_cty = softmax_adjusted_loss(LogitsModel.a_Logits[0],tf.squeeze(A_cty,[1]),ADV,M,ent_coef) * cty_coef
    #loss chat nr
    cnr_coef = 1/5
    loss_cnr = softmax_adjusted_loss(LogitsModel.a_Logits[1],tf.squeeze(A_cnr,[1]),ADV,M,ent_coef) * cnr_coef
    #loss message type
    mty_coef = 1/5
    loss_mty = softmax_adjusted_loss(LogitsModel.a_Logits[2],tf.squeeze(A_mty,[1]),ADV,M,ent_coef) * mty_coef  
    #loss message
    mes_coef = 1/5
    loss_mes = softmax_adjusted_loss_2dim(LogitsModel.a_Logits[3],A_msg,LogitsModel.mask,ADV,M,ent_coef)* mes_coef  
    #loss act
    act_coef = 1/5
    loss_act = sigmoid_adjusted_loss(LogitsModel.a_Logits[4],A_act,ADV,M,ent_coef) * act_coef

    pg_loss = loss_cty + loss_cnr + loss_mty + loss_mes + loss_act

    return pg_loss

def ValueFunction_Loss(model,R):
    vf_loss = tf.reduce_mean(mse(tf.squeeze(model.v_logits), R))
    return vf_loss

def Latent_Loss(model):
    mean = model.e_stats[0]
    log_sigma_sqrt = model.e_stats[1]
    latent_losses = 0.5 * tf.reduce_sum(tf.square(mean) + tf.exp(log_sigma_sqrt) - log_sigma_sqrt - 1,1)
    latent_loss = tf.reduce_mean(latent_losses)
    return latent_loss