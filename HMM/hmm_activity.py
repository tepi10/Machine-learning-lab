# Hidden Markov Model using Viterbi Algorithm
# All inputs are taken from the user

# -------- USER INPUT SECTION --------
n_states = int(input("Enter number of states: "))
states = []
for i in range(n_states):
    states.append(input(f"Enter state {i+1} name: "))

n_obs_types = int(input("\nEnter number of observation types: "))
observations_list = []
for i in range(n_obs_types):
    observations_list.append(input(f"Enter observation type {i+1}: "))

# Initial Probabilities
print("\nEnter Initial Probabilities:")
start_prob = {}
for s in states:
    start_prob[s] = float(input(f"P({s}): "))

# Transition Probabilities
print("\nEnter Transition Probabilities:")
trans_prob = {}
for s in states:
    trans_prob[s] = {}
    for s2 in states:
        trans_prob[s][s2] = float(input(f"P({s} → {s2}): "))

# Emission Probabilities
print("\nEnter Emission Probabilities:")
emit_prob = {}
for s in states:
    emit_prob[s] = {}
    for obs in observations_list:
        emit_prob[s][obs] = float(input(f"P({obs}|{s}): "))

# -------- OBSERVATION SEQUENCE INPUT --------
while True:
    obs_input = input("\nEnter observation sequence separated by space: ")
    observations = obs_input.split()
    
    if all(obs in observations_list for obs in observations):
        break
    else:
        print("❌ Invalid observation entered.")

# -------- MATRIX DISPLAY --------
print("\nInitial Probabilities:")
for s in states:
    print(f"{s}: {start_prob[s]}")

print("\nTransition Probability Matrix:")
print("\t" + "\t".join(states))
for s in states:
    row = [str(trans_prob[s][s2]) for s2 in states]
    print(f"{s}\t" + "\t".join(row))

print("\nEmission Probability Matrix:")
print("\t" + "\t".join(observations_list))
for s in states:
    row = [str(emit_prob[s][o]) for o in observations_list]
    print(f"{s}\t" + "\t".join(row))

# -------- VITERBI ALGORITHM --------
V = [{}]
path = {}

# Initialize
for state in states:
    V[0][state] = start_prob[state] * emit_prob[state][observations[0]]
    path[state] = [state]

print("\nStep-by-Step Viterbi Table")
print(f"\nStep 1 Observation: {observations[0]}")
for state in states:
    print(f"{state}: {V[0][state]:.6f}")

# Recursion
for t in range(1, len(observations)):
    V.append({})
    new_path = {}
    print(f"\nStep {t+1} Observation: {observations[t]}")
    
    for curr_state in states:
        max_prob, prev_state_selected = max(
            (V[t-1][prev_state] *
             trans_prob[prev_state][curr_state] *
             emit_prob[curr_state][observations[t]], prev_state)
            for prev_state in states
        )

        V[t][curr_state] = max_prob
        new_path[curr_state] = path[prev_state_selected] + [curr_state]
        print(f"{curr_state}: {max_prob:.6f} (from {prev_state_selected})")

    path = new_path

# Final result
max_prob, final_state = max((V[-1][state], state) for state in states)
best_path = path[final_state]

print("\nMost likely state sequence:")
print(" → ".join(best_path))
print(f"Final Probability: {max_prob:.6f}")
