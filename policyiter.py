import numpy as np
import scipy.integrate as integrate

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
    def __init__(self,curr_policy,beta,mu,gamma,k_1,k_2,curr_t,next_t,q_i,R_tau):
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

    def to_integrate(self):
       return self.beta * self.Z_tau + self.beta*self.q_i

    def eval(self):
        self.k_3 = self.k_2 - np.log(self.gamma**(-1) * self.beta)
        self.Z_tau = self.k_1 * self.R_tau - self.k_2 * self.q_i + self.k_3 * self.q_i
        self.q_i=integrate.quad(to_integrate,self.curr_t,self.next_t, args=(self.Z_tau))

    def improv(self):
        self.curr_policy=np.argmax(self.q_i)

# testing algorithm
def eps_greedy_policy(q_values, eps):
    probs=np.ones([len(q_values)])*eps/len(q_values)
    best_a=np.argmax(q_values)
    probs[best_a]+=(1-eps)
    policy=probs
    return policy


def q_learning(eps, gamma, mdp):
    Q = np.zeros([16, 4])  # state action value table
    pi = np.zeros([16])  # greedy policy table
    alpha = .01
    # YOUR CODE HERE

    episodes = 10000
    for episode in range(episodes):
        state, reward, terminal = mdp.reset()
        s = state
        # for s in mdp.get_states():
        while terminal == False:
            policy = eps_greedy_policy(Q[s], eps)
            a = np.random.choice(len(policy), p=policy)
            s_next, r, terminal = mdp.step(a)
            Q[s, a] = Q[s, a] + alpha * (r + gamma * np.max(Q[s_next]) - Q[s, a])
            s = s_next
    for s in mdp.get_states():
        pi[s] = np.argmax(Q[s])

policy_iter=IQPI()