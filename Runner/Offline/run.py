import os
import time
import tensorflow as tf
import argparse

from Runner.Offline.runner import Runner
from Model.model import Model




try:
    from SwergioUtility.CommunicationEnviroment import CommunicationEnviroment
except ImportError:
    import pip
    pip.main(['install','-e','/shared/MessageUtilities'])
    time.sleep(10)
    from SwergioUtility.CommunicationEnviroment import CommunicationEnviroment


def run():

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-bs','--batchsize', help='size of batch and steps per learning', type=int, default=20)
    parser.add_argument('-ms','--memorysize', help='size of memory batches', type=int, default=10)
    parser.add_argument('-ls','--latentsize', help='size of latent vektor', type=int, default=30)

    parser.add_argument('-lr','--learningrate', help='learning rate', type=float, default=7e-3)
    parser.add_argument('--lrschedule', help='Learning rate schedule', choices=['constant', 'linear'], default='constant')

    parser.add_argument('--alpha', help='alpha', type=float, default=0.99)
    parser.add_argument('--epsilon', help='epsilon', type=float, default=1e-5)
    parser.add_argument('--maxgradnorm', help='max_grad_norm', type=float, default=0.5)

    parser.add_argument('--entcoef', help='entropy coefficient', type=float, default=0.01)
    
    parser.add_argument('--latentlossweight', help='weight latent loss', type=float, default=1)
    parser.add_argument('--generationlossweight', help='weight generation loss', type=float, default=1)
    parser.add_argument('--GANGlossweight', help='weight GAN G loss', type=float, default=1)
    parser.add_argument('--GANDlossweight', help='weight GAM D loss', type=float, default=1)
    parser.add_argument('--policylossweight', help='weight policy loss', type=float, default=1)
    parser.add_argument('--criticlossweight', help='weight critic loss', type=float, default=0.5)

    parser.add_argument('-ts','--trainsteps', help='steps of training', type=int, default=50000)
    parser.add_argument('-li','--logint', help='log intervall', type=int, default=500)
    parser.add_argument('-si','--saveint', help='save intervall', type=int, default=5000)

    parser.add_argument('--logdir', help='path to log directory')
    parser.add_argument('--savedir', help='path to save directory')

    parser.add_argument('--expertdata', help='path to expertdata file')

    parser.add_argument('--agents', help='set the agent names', nargs = '*')
    parser.add_argument('--worker', help='set the worker names', nargs = '*')
    parser.add_argument('--knowledge', help='set the knowledgebase names', nargs = '*')

    args = parser.parse_args()

    batch_size  = args.batchsize 
    memory_size = args.memorysize
    latent_size = args.latentsize

    lr = args.learningrate
    lrschedule = args.lrschedule

    alpha=args.alpha
    epsilon = args.epsilon
    max_grad_norm = args.maxgradnorm
    ent_coef = args.entcoef

    latent_loss_weight = args.latentlossweight
    generation_loss_weight = args.generationlossweight
    GAN_G_loss_weight = args.GANGlossweight
    GAN_D_loss_weight = args.GANDlossweight
    policy_loss_weight = args.policylossweight
    critic_loss_weight = args.criticlossweight

    train_steps = args.trainsteps
    log_interval= args.logint
    save_interval = args.saveint

    expertdata_path = args.expertdata

    agents = args.agents if args.agents is not None else []
    worker = args.worker if args.worker is not None else []
    knowledge = args.knowledge if args.knowledge is not None else []

    if args.logdir is None:
        logpath = os.getenv('LOG_PATH')
    else:
        logpath = args.logdir

    if args.savedir is None:
        modelsavepath = os.getenv('MODELSAVE_PATH')
    else:
        modelsavepath = args.savedir

    _SocketIONamespaces = {'AGENT':agents,'WORKER':worker, 'KNOWLEDGEBASE':knowledge}
    env = CommunicationEnviroment(SocketIONamespaces = _SocketIONamespaces, no_socketIO_client = True,expertdata_file_path = expertdata_path)


    tf.reset_default_graph()

    model = Model(env=env, 
        batch_size=batch_size, 
        memory_size = memory_size, 
        latent_size = latent_size,
        ent_coef=ent_coef, 
        max_grad_norm=max_grad_norm, 
        alpha=alpha,
        epsilon=epsilon,
        latent_loss_weight = latent_loss_weight,
        generation_loss_weight =generation_loss_weight,
        GAN_G_loss_weight = GAN_G_loss_weight,
        GAN_D_loss_weight = GAN_D_loss_weight,
        policy_loss_weight = policy_loss_weight,
        critic_loss_weight = critic_loss_weight )

    runner = Runner(env, model,
        modelsavepath = modelsavepath,
        logpath = logpath,
        lr = lr,
        train_steps=train_steps,
        log_interval = log_interval, 
        save_interval = save_interval )
    
    time.sleep(1)
    runner.train()


if __name__ == '__main__':
    run()


