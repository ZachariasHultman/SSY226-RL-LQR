import numpy as np
import scipy.integrate as integrate
import torch
from collections import deque

#Implementing algoritm 3a from "integral policy interation".
#curr_policy: current policy Pi_i
#beta: weighting factor
#mu:Action dependent policy
#gamma:Discount factor
#k_1: nonzero constant
#k_2: nonzero constant
#q_i: q-function dependent on state x and policy (also dependent on x)
#R_tau: reward

class IQPI:
    def __init__(self,curr_policy,beta,mu,gamma,k_1,k_2,curr_t,next_t,q_i,R_tau,tau,Z_tau):
        self.curr_policy=curr_policy
        self.beta=beta
        self.mu=mu
        self.gamma=gamma
        self.k_1=k_1
        self.k_2=k_2
        self.curr_t=curr_t
        self.next_t=next_t
        self.q_i=q_i
        self.R_tau=R_tau
        self.tau=tau
        self.Z_tau=Z_tau

    def to_integrate(self,beta,tau,curr_t,Z_tau):
        self.integrand = beta ** (tau - curr_t) * Z_tau
        return self.integrand       #+ self.beta * self.q_i
       #return self.beta * self.Z_tau + self.beta*self.q_i

    def eval(self,k_1,k_2,gamma,beta,R_tau,q_i,curr_t,next_t,integrand):
        self.k_3 = k_2 - np.log(gamma**(-1) * beta)
        self.Z_tau = k_1 * R_tau - k_2 * q_i + self.k_3 * q_i
        self.q_i=integrate.quad(integrand,curr_t,next_t, args=(self.Z_tau,curr_t))+beta * q_i
        return self.q_i, self.Z_tau, self.k_3

    def improv(self,q_i):
        self.curr_policy=np.argmax(q_i)
        return self.curr_policy

# testing algorithm below

def eps_greedy_policy(q_values, eps):
    probs=np.ones([len(q_values)])*eps/len(q_values)
    best_a=np.argmax(q_values)
    probs[best_a]+=(1-eps)
    policy=probs
    return policy


def cart_pendulum_sim(t , x, L=1., m=1., M = 1., g=9.81, F=0, f=0,f_cart=0):
    """
    x_1_dot = velocity
    x_2_dot = acceleration
    x_3_dot = angular velocity
    x_4_dot = angular acceleration
    """
    # print(x)
    x1, x2, x3, x4 = x
    x_2_dot_nomi = (-m*g*np.sin(x3)*np.cos(x3) +
                    m*L*x4*x4*np.sin(x3) +
                    f*m*x4*np.cos(x3)+F)-x2*f_cart

    x_2_dot_denomi = (M + (1 - np.cos(x3) * np.cos(x3)) * m)

    x_4_dot_nomi = ((M+m)*(g*np.sin(x3)-f*x4) -
                    (L*m*x4*x4*np.sin(x3)+F) * np.cos(x3))

    x_4_dot_denomi = L*x_2_dot_denomi

    x_1_dot = x2
    x_2_dot = x_2_dot_nomi/x_2_dot_denomi
    x_3_dot = x4
    x_4_dot = x_4_dot_nomi/x_4_dot_denomi ###

    return [x_1_dot, x_2_dot, x_3_dot, x_4_dot]

class ExperienceReplay:
    def __init__(self, device, num_states, buffer_size=1e+6):
        self._device = device
        self.__buffer = deque(maxlen=int(buffer_size))
        self._num_states = num_states

    @property
    def buffer_length(self):
        return len(self.__buffer)

    def add(self, transition):
        '''
        Adds a transition <s, a, r, s', t > to the replay buffer
        :param transition:
        :return:
        '''
        self.__buffer.append(transition)

    def sample_minibatch(self, batch_size=128):
        '''
        :param batch_size:
        :return:
        '''
        ids = np.random.choice(a=self.buffer_length, size=batch_size)
        state_batch = np.zeros([batch_size, self._num_states],
                               dtype=np.float32)
        action_batch = np.zeros([
            batch_size,
        ], dtype=np.int64)
        reward_batch = np.zeros([
            batch_size,
        ], dtype=np.float32)
        nonterminal_batch = np.zeros([
            batch_size,
        ], dtype=np.bool)
        next_state_batch = np.zeros([batch_size, self._num_states],
                                    dtype=np.float32)
        for i, index in zip(range(batch_size), ids):
            state_batch[i, :] = self.__buffer[index].s
            action_batch[i] = self.__buffer[index].a
            reward_batch[i] = self.__buffer[index].r
            nonterminal_batch[i] = self.__buffer[index].t
            next_state_batch[i, :] = self.__buffer[index].next_s

        return (
            torch.tensor(state_batch, dtype=torch.float, device=self._device),
            torch.tensor(action_batch, dtype=torch.long, device=self._device),
            torch.tensor(reward_batch, dtype=torch.float, device=self._device),
            torch.tensor(next_state_batch,
                         dtype=torch.float,
                         device=self._device),
            torch.tensor(nonterminal_batch,
                         dtype=torch.bool,
                         device=self._device),
        )

t=20
x=1,1,1,1
x1dot,x2dot,x3dot,x4dot=cart_pendulum_sim(t , x, L=1., m=1., M = 1., g=9.81, F=0, f=0,f_cart=0) #is working

eps=0.1
q_values=np.ones([4,2])
curr_policy=eps_greedy_policy(q_values,eps)
print('curr_pol',curr_policy)
beta=0.7
mu=curr_policy
gamma=1
k_1=1
k_2=1
curr_t=1
next_t=2
q_i=np.ones([4,2])
R_tau=np.ones([4,1])
Z_tau=1
tau=1

device=device = torch.device("cpu")
replay_buffer=ExperienceReplay(device, len(x), buffer_size=1e+6) #creating replay buffer

if k_1>0 and k_2>0:
    policy_iter=IQPI(curr_policy,beta,mu,gamma,k_1,k_2,curr_t,next_t,q_i,R_tau,tau,Z_tau) #is working
    #print(policy_iter.improv())
    integrand=policy_iter.to_integrate(beta,tau,curr_t,Z_tau)
    eval_pol=policy_iter.eval(k_1,k_2,gamma,beta,R_tau,q_i,curr_t,next_t,integrand) #def eval(self,k_1,k_2,gamma,beta,R_tau,q_i,curr_t,next_t,integrand):
    improv_pol=policy_iter.improv(q_i)
#Improving policy until it becomes optimal
    while policy_iter.curr_policy!=all(improv_pol):
        policy_iter = IQPI(curr_policy, beta, mu, gamma, k_1, k_2, curr_t, next_t, q_i, R_tau,tau,Z_tau)
        curr_t=next_t
        next_t=next_t+1
        improv_pol = policy_iter.improv()













#class QLearningModel(object):
#    def __init__(self, device, num_states, num_actions, learning_rate):
#        self._device = device
#        self._num_states = num_states
#        self._num_actions = num_actions
#        self._lr = learning_rate

        # Define the two deep Q-networks
        #self.online_model = QNetwork(self._num_states,
        #                             self._num_actions).to(device=self._device)
        #self.offline_model = QNetwork(
        #    self._num_states, self._num_actions).to(device=self._device)

        # Define optimizer. Should update online network parameters only.
        #self.optimizer = torch.optim.RMSprop(self.online_model.parameters(),
        #                                     lr=self._lr)

        # Define loss function
        #self._mse = nn.MSELoss(reduction='mean').to(device=self._device)





#def train_loop_ddqn(ddqn, env, replay_buffer, num_episodes, enable_visualization=False, batch_size=64, gamma=.94):
#    Transition = namedtuple("Transition", ["s", "a", "r", "next_s", "t"])
#    eps = 1.
#    eps_end = .1
#    eps_decay = .001
#    tau = 1000
#    cnt_updates = 0
#    R_buffer = []
#    R_avg = []
#    for i in range(num_episodes):
#        state = env.reset()  # Initial state
#        state = state[None, :]  # Add singleton dimension, to represent as batch of size 1.
#        finish_episode = False  # Initialize
#        ep_reward = 0  # Initialize "Episodic reward", i.e. the total reward for episode, when disregarding discount factor.
#        q_buffer = []
#        steps = 0
#        while not finish_episode:
#            if enable_visualization:
#                env.render()  # comment this line out if you don't want to / cannot render the environment on your system
#            steps += 1

            # Take one step in environment. No need to compute gradients,
            # we will just store transition to replay buffer, and later sample a whole batch
            # from the replay buffer to actually take a gradient step.
#            q_online_curr, curr_action = calc_q_and_take_action(ddqn, state, eps)
#            q_buffer.append(q_online_curr)
#            new_state, reward, finish_episode, _ = env.step(curr_action)  # take one step in the evironment
#            new_state = new_state[None, :]

            # Assess whether terminal state was reached.
            # The episode may end due to having reached 200 steps, but we should not regard this as reaching the terminal state, and hence not disregard Q(s',a) from the Q target.
            # https://arxiv.org/abs/1712.00378
#            nonterminal_to_buffer = not finish_episode or steps == 200

            # Store experienced transition to replay buffer
#            replay_buffer.add(Transition(s=state, a=curr_action, r=reward, next_s=new_state, t=nonterminal_to_buffer))

#            state = new_state
#            ep_reward += reward

            # If replay buffer contains more than 1000 samples, perform one training step
#            if replay_buffer.buffer_length > 1000:
#                loss = sample_batch_and_calculate_loss(ddqn, replay_buffer, batch_size, gamma)
#                ddqn.optimizer.zero_grad()
#                loss.backward()
#                ddqn.optimizer.step()

#                cnt_updates += 1
#                if cnt_updates % tau == 0:
#                    ddqn.update_target_network()

#        eps = max(eps - eps_decay, eps_end)  # decrease epsilon
#        R_buffer.append(ep_reward)

        # Running average of episodic rewards (total reward, disregarding discount factor)
#        R_avg.append(.05 * R_buffer[i] + .95 * R_avg[i - 1]) if i > 0 else R_avg.append(R_buffer[i])

#        print('Episode: {:d}, Total Reward (running avg): {:4.0f} ({:.2f}) Epsilon: {:.3f}, Avg Q: {:.4g}'.format(i,
#                                                                                                                  ep_reward,
#                                                                                                                  R_avg[
#                                                                                                                      -1],
#                                                                                                                  eps,
#                                                                                                                  np.mean(
#                                                                                                                      np.array(
#                                                                                                                          q_buffer))))

        # If running average > 195 (close to 200), the task is considered solved
 #       if R_avg[-1] > 195:
  #          return R_buffer, R_avg
  #  return R_buffer, R_avg

#device = torch.device("cpu")
#replay_buffer = ExperienceReplay(device, len(x))

# Create the environment

#env = gym.make("CartPole-v0")

# Enable visualization? Does not work in all environments.
#enable_visualization = False

# Initializations
#num_actions = env.action_space.n
#num_states = env.observation_space.shape[0]
#num_episodes = 1200
#batch_size = 128
#gamma = .94
#learning_rate = 1e-4

# Object holding our online / offline Q-Networks
#ddqn = DoubleQLearningModel(device, num_states, num_actions, learning_rate)

# Create replay buffer, where experience in form of tuples <s,a,r,s',t>, gathered from the environment is stored
# for training
#replay_buffer = ExperienceReplay(device, num_states)

# Train
#R, R_avg = train_loop_ddqn(ddqn, env, replay_buffer, num_episodes, enable_visualization=enable_visualization, batch_size=batch_size, gamma=gamma)
