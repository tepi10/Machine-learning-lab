import numpy as np
import matplotlib.pyplot as plt

# States and observations
states = ["Walking", "Running", "Sitting"]
observations = ["Low", "Medium", "High"]

# Mapping
state_map = {s:i for i,s in enumerate(states)}
obs_map = {o:i for i,o in enumerate(observations)}

# Transition Probabilities
A = np.array([
    [0.6, 0.3, 0.1],
    [0.4, 0.5, 0.1],
    [0.2, 0.1, 0.7]
])

# Emission Probabilities
B = np.array([
    [0.2, 0.6, 0.2],
    [0.1, 0.3, 0.6],
    [0.7, 0.2, 0.1]
])

# Initial Probabilities
pi = np.array([0.5, 0.3, 0.2])

# Viterbi Algorithm
def viterbi(obs_seq, A, B, pi):
    n_states = A.shape[0]
    T = len(obs_seq)

    dp = np.zeros((n_states, T))
    ptr = np.zeros((n_states, T), dtype=int)

    # Initialization
    dp[:, 0] = pi * B[:, obs_seq[0]]

    # Recursion
    for t in range(1, T):
        for s in range(n_states):
            prob = dp[:, t-1] * A[:, s] * B[s, obs_seq[t]]
            ptr[s, t] = np.argmax(prob)
            dp[s, t] = np.max(prob)

    # Backtracking
    best_path = np.zeros(T, dtype=int)
    best_path[T-1] = np.argmax(dp[:, T-1])

    for t in range(T-2, -1, -1):
        best_path[t] = ptr[best_path[t+1], t+1]

    return best_path

# Observation sequence
obs_sequence = ["Low", "Medium", "High", "High", "Medium", "Low", "Low"]
obs_seq = [obs_map[o] for o in obs_sequence]

# Run model
path = viterbi(obs_seq, A, B, pi)
predicted_states = [states[i] for i in path]

print("Observations:", obs_sequence)
print("Predicted States:", predicted_states)

# Accuracy check
true_states = ["Sitting", "Walking", "Running", "Running", "Walking", "Sitting", "Sitting"]

correct = sum(p == t for p, t in zip(predicted_states, true_states))
accuracy = correct / len(true_states)

print("Accuracy:", accuracy)

# Visualization
state_numeric = [state_map[s] for s in predicted_states]

plt.plot(state_numeric, marker='o')
plt.yticks(range(len(states)), states)
plt.title("Predicted Activity Over Time")
plt.xlabel("Time Step")
plt.ylabel("State")
plt.grid()
plt.show()