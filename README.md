# A2CGAN Agent

This is an implementation of a swergio agent based on the A2C reinforcement learning model including a VAE-GAN approach and a MemN2N network.


## Model
The model consists of four main components. Similar to the VAE-GAN approach three components are the Encoder, Generator, and Discriminator.  Additional the Critic is implemented for reinforcement learning with the A2C approach.

### Encoder:
The encoders objective is to transform the input information (current and historic Questions & Answers ) to a latent representation.
To reflect the information of prior messages the encoder is using a MemN2N model with an attention mechanism.  

#### Generator
The generator generates the five required information of a future action/ message based on the provided latent vector representation of the current state.
- Act Flag: Decide if the Agent should act based on the input.
- Chat Type: Decide what type of chat (e.g. AGENT, KNOWLEDGEBASE...) the action/message should be sent to
- Chat Number: Decide to which chat of the selected types the action/message should be sent
- Message Type: Decide what type of message (e.g. QUESTION, ANSWER) the agent want to sent
- Message Text:  Generate the action/message text
To generate the message text we use a recurrent neural net with GRU cells.

#### Discriminator
The discriminator is trained to distinct between "real" action/messages (expert data) and "fake" (generated). 
The input for the discriminator is given by the latent vector of the current state (history + last messages) and the suggested action/message.
To optimize the discriminator the loss function of a "least squares GAN"  is used.
To avoid a non-differentiable network, the generator receives the discriminator feedback as a reward and optimize its policy accordingly.

#### Critic
The critic approximates the value function of the current state. It receives the latent vector from the encoder.
The critic is only used while running the model online mode.

## Training and Running

The agent can be trained offline (using "expert data" as ground truth) or online (using firsthand feedback of other swergio components).

#### Offline
In offline mode, the model is trained based on given expert data. This includes the encoder, the generator as well as the discriminator. To fit the discriminator to "fake" data, the generator generates messages based on a random latent vector input. See the chart for an overview:

![Offline Model](/readmesrc/OfflineModel.jpg)

To start the offline training we can run the "runOffline.py" file.

#### Online

In online mode, the model is trained based on the received feedback of connected swergio components. 
The online runner is separated into three main modules.

The actor will receive the last observation and uses the network to generate the next action. To further explore the observation state the actor will randomly choose an action that is not necessarily optimized.

Up to N of the last observations and final actions of the actor are stored in the experience class.

After N steps or end of an episode (next question from a client), the trainer trains the model based on the saved experiences.
In addition to the experience, expert data is used to continuously train the generator and discriminator on "real" data.

See the chart for an overview:

![Online Model](/readmesrc/OnlineModel.jpg)

To start the online mode run the "run.py" file.


## Settings

#### Basic Settings

###### Path to store log files 
Argument: --logdir  
Enviroment variable: LOG_PATH
###### Path to store model saves
Argument: --savedir  
Enviroment variable: MODELSAVE_PATH

###### Path to experdata file: 
Argument: --expertdata  
Enviroment variable: EXPERTDATA_PATH

###### Names of agent namespaces
Argument: --agents (e.g. = agent_A, agent_B)
Enviroment variable: AGENT
###### Names of worker namespaces
Argument: --worker (e.g. = worker_A, worker_B)
Enviroment variable: WORKER
###### Names of knowledgebase namespaces
Argument: --knowledge (e.g. = kb_A, kb_B)
Enviroment variable: KNOWLEDGEBASE


#### Model Settings 

######  Batchsize / expierence steps per learning 
Argument: --batchsize  = 20
######  Size of MemN2N memory
Argument: --memorysize = 10
######  Size of latent vektor
Argument: --latentsize = 30


#### Training Settings

###### Log intervall in training
Argument: -li,--logint =500
###### Save intervall in training
Argument: -si,--saveint =5000

###### Coefficient of entropy in loss function 
Argument: --entcoef = 0.01
###### Weighting of latent losses
Argument: --latentlossweight = 1
###### Weighting of generation losses
Argument: --generationlossweight = 1
###### Weighting of GAN Generator losses
Argument: --GANGlossweight = 1
###### Weighting of GAN Discriminator losses
Argument: --GANDlossweight = 1
###### Weighting of policy losses
Argument: --policylossweight = 1
###### Weighting of critc losses
Argument: --criticlossweight = 0.5
###### Decay of RMSPropPtimizer
Argument: --alpha = 0.99
###### Epsilon of RMSPropPtimizer
Argument: --epsilon = 1e-5
###### Maximum gradient norm
Argument: --maxgradnorm = 0.5
###### Learning Rate
Argument: -lr,--learningrate =7e-3

#### Additional Settings for Offline Training

###### Number of overall trainig steps
Argument: -ts,trainsteps-- =50000

#### Additional Settings for Online Mode

###### Discounting factor of rewards
Argument: --gamma =0.99
###### Lamda factor of discounting in generelized advantage estimate
Argument:  --gaelambda =0.96
###### Probability of actor to perform an action (1 -> always, 0 -> never)
Argument:  --actprobability =1
###### Probability of actor to explor te action space (randomly choose action)
Argument:  --explorprobability =0.05

## References
This is an overview of the main sources and ideas being used for code snippets, implementation, and inspiration of this model.

###### A2C Model
Publication - https://arxiv.org/abs/1602.01783
Example implementation in Open AI's baselines library - https://github.com/openai/baselines

###### Least Squares GAN (LSGAN)
Publication - https://arxiv.org/abs/1611.04076v2
Blog Post with explanation and implementation from Augustin Kristiadi - https://wiseodd.github.io/techblog/2017/03/02/least-squares-gan/

###### MemN2N Model
Publication - https://arxiv.org/abs/1503.08895v4
Example implementation -https://github.com/carpedm20/MemN2N-tensorflow

###### VAE-GAN
Publication - https://arxiv.org/pdf/1512.09300.pdf
Example implentation - https://github.com/timsainb/Tensorflow-MultiGPU-VAE-GAN 

###### Seq2Seq Model 
Explanation and example implementation in tutorial for  tensorflow - https://github.com/tensorflow/nmt




