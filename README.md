# POLICY EVALUATION

## AIM
To develop a Python program to evaluate the given policy.

## PROBLEM STATEMENT

The bandit slippery walk problem is a reinforcement learning problem in which an agent must learn to navigate a 7-state environment in order to reach a goal state. The environment is slippery, so the agent has a chance of moving in the opposite direction of the action it takes.

### States
The environment has 7 states:

- Two Terminal States: 
    - G: The goal state  
    - H: A hole state.
- Five Transition states / Non-terminal States including S: The starting state.
### Actions
The agent can take two actions:

- R: Move right.
- L: Move left.

### Transition Probabilities
The transition probabilities for each action are as follows:

- 50% chance that the agent moves in the intended direction.
- 33.33% chance that the agent stays in its current state.
- 16.66% chance that the agent moves in the opposite direction.

For example, if the agent is in state S and takes the "R" action, then there is a 50% chance that it will move to state 4, a 33.33% chance that it will stay in state S, and a 16.66% chance that it will move to state 2.

### Rewards
- The agent receives a reward of +1 for reaching the goal state (G). 
- The agent receives a reward of 0 for all other states.

### Graphical Representation
![image](https://github.com/Venkatigi/rl-policy-evaluation/assets/94154252/920712aa-ba9f-4520-95b4-e6b2c2e27efc)

## POLICY EVALUATION FUNCTION
![image](https://github.com/Venkatigi/rl-policy-evaluation/assets/94154252/5b35226f-6fa5-4f53-9d4d-6a086fdc8b16)

## PROGRAM
```py
#program developed by : SRIJITH R
#register no: 212221240054
```

### predefined environment and functions:
```py
pip install git+https://github.com/mimoralea/gym-walk#egg=gym-walk

import warnings ; warnings.filterwarnings('ignore')
import gym, gym_walk
import numpy as np
import random
import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)
np.set_printoptions(suppress=True)
random.seed(123); np.random.seed(123)
```

```py
def print_policy(pi, P, action_symbols=('<', 'v', '>', '^'), n_cols=4, title='Policy:'):
    print(title)
    arrs = {k:v for k,v in enumerate(action_symbols)}
    for s in range(len(P)):
        a = pi(s)
        print("| ", end="")
        if np.all([done for action in P[s].values() for _, _, _, done in action]):
            print("".rjust(9), end=" ")
        else:
            print(str(s).zfill(2), arrs[a].rjust(6), end=" ")
        if (s + 1) % n_cols == 0: print("|")


def print_state_value_function(V, P, n_cols=4, prec=3, title='State-value function:'):
    print(title)
    for s in range(len(P)):
        v = V[s]
        print("| ", end="")
        if np.all([done for action in P[s].values() for _, _, _, done in action]):
            print("".rjust(9), end=" ")
        else:
            print(str(s).zfill(2), '{}'.format(np.round(v, prec)).rjust(6), end=" ")
        if (s + 1) % n_cols == 0: print("|")


def probability_success(env, pi, goal_state, n_episodes=100, max_steps=200):
    random.seed(123); np.random.seed(123) ; env.seed(123)
    results = []
    for _ in range(n_episodes):
        state, done, steps = env.reset(), False, 0
        while not done and steps < max_steps:
            state, _, done, h = env.step(pi(state))
            steps += 1
        results.append(state == goal_state)
    return np.sum(results)/len(results)


def mean_return(env, pi, n_episodes=100, max_steps=200):
    random.seed(123); np.random.seed(123) ; env.seed(123)
    results = []
    for _ in range(n_episodes):
        state, done, steps = env.reset(), False, 0
        results.append(0.0)
        while not done and steps < max_steps:
            state, reward, done, _ = env.step(pi(state))
            results[-1] += reward
            steps += 1
    return np.mean(results)


env = gym.make('SlipperyWalkFive-v0')
P = env.env.P
init_state = env.reset()
goal_state = 6
LEFT, RIGHT = range(2)
```

### policy evaluation: 
```py
def policy_evaluation(pi, P, gamma=1.0, theta=1e-10):
    prev_V = np.zeros(len(P), dtype=np.float64)
    while True:
      V=np.zeros(len(P),dtype=np.float64)
      for s in range(len(P)):
        for prob, next_state, reward, done in P[s][pi(s)]:
          V[s]+=prob*(reward+gamma*prev_V[next_state]*(not done))
      if np.max(np.abs(prev_V-V))<theta:
        break
      prev_V=V.copy()
    return V

# First Policy
pi_1 = lambda s: {
    0:LEFT, 1:LEFT, 2:LEFT, 3:LEFT, 4:LEFT, 5:LEFT, 6:LEFT
}[s]
print_policy(pi_1, P, action_symbols=('<', '>'), n_cols=7)

# Second Policy
pi_2 = lambda s: {
    0:RIGHT, 1:RIGHT, 2:LEFT, 3:RIGHT, 4:LEFT, 5:RIGHT, 6:RIGHT
}[s]
print_policy(pi_2, P, action_symbols=('<', '>'), n_cols=7)
```
### policy comparison:
```py 
# Code to evaluate the first policy
V1 = policy_evaluation(pi_1, P)
print(V1)
print_state_value_function(V1, P, n_cols=7, prec=5)

# Code to evaluate the second policy
V2 = policy_evaluation(pi_2, P)
print(V2)
print_state_value_function(V2, P, n_cols=7, prec=5)

# Compare the two policies based on the value function using the above equation and find the best policy
print(V1>=V2)
if(np.sum(V1>=V2)==7):
  print("The first policy is the better policy")
elif(np.sum(V2>=V1)==7):
  print("The second policy is the better policy")
else:
  print("Both policies have their merits.")
```

## OUTPUT:
## <u> policy 1: </u>

### policy
![image](https://github.com/Venkatigi/rl-policy-evaluation/assets/94154252/7032868e-c983-4fa5-9a1c-b876ec05e52d)

### policy evaluation:
![image](https://github.com/Venkatigi/rl-policy-evaluation/assets/94154252/a3b3f7a4-fbef-4378-b639-3fa4863d3b5d)

### state value function:
![image](https://github.com/Venkatigi/rl-policy-evaluation/assets/94154252/31b10d69-fc5c-4b21-99e4-d311ac4e6924)

## <u> policy 2: </u>
### policy
![image](https://github.com/Venkatigi/rl-policy-evaluation/assets/94154252/e653eb7c-4ea5-44cf-8f69-fb77584b9d40)

### policy evaluation:
![image](https://github.com/Venkatigi/rl-policy-evaluation/assets/94154252/45bdfaed-1fb1-41f5-bfb6-a40313add9b0)

### state value function:
![image](https://github.com/Venkatigi/rl-policy-evaluation/assets/94154252/3eb368da-2506-4ca4-9c9f-3a1947d0a322)

## <u> policy comparison: </u> 
![image](https://github.com/Venkatigi/rl-policy-evaluation/assets/94154252/26c8edbc-e35a-4842-b2bb-2dafcbdf574a)

## RESULT:
Thus, a Python program is developed to evaluate the given policy.
