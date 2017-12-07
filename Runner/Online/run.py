import os
import time
import tensorflow as tf
import argparse

from Model.model import Model
from Runner.Online.runner import Runner



try:
    from SwergioUtility.CommunicationEnviroment import CommunicationEnviroment
except ImportError:
    import pip
    pip.main(['install','-e','/shared/MessageUtilities'])
    time.sleep(10)
    from SwergioUtility.CommunicationEnviroment import CommunicationEnviroment


def run():

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-bs','--batchsize', help='size of batch and steps per learning', type=int, default=5)
    parser.add_argument('-ms','--memorysize', help='size of memory batches', type=int, default=10)
    parser.add_argument('-ls','--latentsize', help='size of latent vektor', type=int, default=30)

    parser.add_argument('-lr','--learningrate', help='learning rate', type=float, default=7e-4)
    parser.add_argument('--lrschedule', help='Learning rate schedule', choices=['constant', 'linear'], default='constant')

    parser.add_argument('--alpha', help='alpha', type=float, default=0.99)
    parser.add_argument('--gamma', help='gamma', type=float, default=0.99)
    parser.add_argument('--gaelambda', help='lambda for genralized advatage estimate', type=float, default=0.96)
    parser.add_argument('--epsilon', help='epsilon', type=float, default=1e-5)
    parser.add_argument('--maxgradnorm', help='max_grad_norm', type=float, default=0.5)

    parser.add_argument('--actprobability', help='Probability of the actor to act', type=float, default=1)
    parser.add_argument('--explorprobability', help='Probability of the actor to explor', type=float, default=0.05)



    parser.add_argument('--entcoef', help='entropy coefficient', type=float, default=0.01)
    
    parser.add_argument('--latentlossweight', help='weight latent loss', type=float, default=1)
    parser.add_argument('--generationlossweight', help='weight generation loss', type=float, default=1)
    parser.add_argument('--GANGlossweight', help='weight GAN G loss', type=float, default=1)
    parser.add_argument('--GANDlossweight', help='weight GAM D loss', type=float, default=1)
    parser.add_argument('--policylossweight', help='weight policy loss', type=float, default=1)
    parser.add_argument('--criticlossweight', help='weight critic loss', type=float, default=0.5)

    parser.add_argument('-li','--logint', help='log intervall', type=int, default=100)
    parser.add_argument('-si','--saveint', help='save intervall', type=int, default=500)

    parser.add_argument('--logdir', help='path to log directory')
    parser.add_argument('--savedir', help='path to save directory')

    parser.add_argument('--expertdata', help='path to expertdata file')

    args = parser.parse_args()

    batch_size  = args.batchsize 
    memory_size = args.memorysize
    latent_size = args.latentsize

    lr = args.learningrate
    lrschedule = args.lrschedule

    alpha=args.alpha
    gamma = args.gamma
    gae_lambda = args.gaelambda
    epsilon = args.epsilon
    max_grad_norm = args.maxgradnorm
    ent_coef = args.entcoef

    latent_loss_weight = args.latentlossweight
    generation_loss_weight = args.generationlossweight
    GAN_G_loss_weight = args.GANGlossweight
    GAN_D_loss_weight = args.GANDlossweight
    policy_loss_weight = args.policylossweight
    critic_loss_weight = args.criticlossweight

    log_interval= args.logint
    save_interval = args.saveint

    if args.logdir is None:
        logpath = os.getenv('LOG_PATH')
    else:
        logpath = args.logdir

    if args.savedir is None:
        modelsavepath = os.getenv('MODELSAVE_PATH')
    else:
        modelsavepath = args.savedir

    if args.expertdata is None:
        expertdata_path = os.getenv('EXPERTDATA_PATH')
    else:
        expertdata_path = args.expertdata

    env = CommunicationEnviroment(expertdata_file_path = expertdata_path)

    tf.reset_default_graph()
    print('load model')
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

    print('load runner')
    runner = Runner(env, model,
        modelsavepath = modelsavepath,
        logpath = logpath,
        lr = lr,
        gamma = gamma,
        gae_lambda = gae_lambda,
        log_interval = log_interval, 
        save_interval = save_interval )
    
    time.sleep(1)
    print('start')
    runner.listen()


if __name__ == '__main__':
    run()


