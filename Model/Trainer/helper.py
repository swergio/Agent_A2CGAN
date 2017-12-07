import tensorflow as tf

def RMSPropTrainer(loss, params, LR, alpha , epsilon, max_grad_norm, name):
    grads = tf.gradients(loss, params)
    if max_grad_norm is not None:
        grads, _ = tf.clip_by_global_norm(grads, max_grad_norm)
    grads = list(zip(grads, params))
    trainer = tf.train.RMSPropOptimizer(learning_rate=LR, decay=alpha, epsilon=epsilon, name = name)
    _train = trainer.apply_gradients(grads)
    return _train, grads