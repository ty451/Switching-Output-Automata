import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Union, Optional
from dataclasses import dataclass
import random
import matplotlib.pyplot as plt

@dataclass
class SOA:
    X: List[str]  # Set of states
    Y: List[int]  # Set of outputs
    h: Dict[str, List[int]]  # State output mapping function
    B: List[Tuple[str, str]]  # Set of edges
    x0: str  # Initial state
    y0: int  # Initial output
    ss: Dict[str, List[int]]  # Secret state mapping
    SecretDwellTime: Dict[str, List[List[Union[int, float]]]]  # Secret dwell time intervals

@dataclass
class EvolutionAutomaton:
    Q: List[str]  # States
    Ye: List[Union[int, str]]  # Events
    Delta: List[Tuple[str, str, str]]  # Transitions
    q0: str  # Initial state
    sls: List[str]  # Secret logical states

@dataclass
class Observer:
    Z: List[List[str]]  # States
    Ye: List[Union[int, str]]  # Events
    Delta_o: List[str]  # Transitions
    z0: List[str]  # Initial state

def generate_global_states(h: Dict[str, List[int]]) -> List[str]:
    """Generate all possible global states from the state output mapping."""
    global_states = []
    for state, outputs in h.items():
        for output in outputs:
            global_states.append(f"({state},{output})")
    return global_states

def get_upper_limit(G: SOA, global_states: List[str]) -> Dict[str, int]:
    """Calculate the upper limit for each global state."""
    max_mapping = {}

    # Process SecretDwellTime entries
    for key, value_sets in G.SecretDwellTime.items():
        max_value = 1  # Default value
        for interval in value_sets:
            finite_values = [v for v in interval if v != float('inf')]
            if finite_values:
                max_value = max(max_value, max(finite_values))
        max_mapping[key] = max_value

    # Process remaining global states
    for state in global_states:
        if state not in max_mapping:
            max_mapping[state] = 1

    return max_mapping

def sigma(x: str, B: List[Tuple[str, str]]) -> List[str]:
    """Find all states that can be reached from state x."""
    return [b[1] for b in B if b[0] == x]

def D_epsilon(q: str, Ge: EvolutionAutomaton) -> List[str]:
    """Compute epsilon closure of state q."""
    D_epsilon_q = [q]
    queue = [q]

    while queue:
        current_state = queue.pop(0)
        epsilon_transitions = [(i, t) for i, t in enumerate(Ge.Delta)
                               if t[0] == current_state and t[1] == 'epsilon']

        for _, transition in epsilon_transitions:
            next_state = transition[2]
            if next_state not in D_epsilon_q:
                D_epsilon_q.append(next_state)
                queue.append(next_state)

    return D_epsilon_q

def D_y(q: str, y: Union[int, str], Ge: EvolutionAutomaton) -> List[str]:
    """Compute states reachable from q on input y."""
    return [t[2] for t in Ge.Delta if t[0] == q and t[1] == str(y)]

def parse_state(state_str: str) -> Tuple[str, str, str]:
    """Parse a state string into components."""
    import re
    match = re.match(r'\((.*?),(.*?)\)(.*)', state_str)
    if match:
        return match.group(1), match.group(2), match.group(3)
    raise ValueError(f"Invalid state string format: {state_str}")

def construct_evolution_automaton(G: SOA) -> EvolutionAutomaton:
    """Construct the evolution automaton from the SOA."""
    # Initialize variables
    global_states = generate_global_states(G.h)
    max_mapping = get_upper_limit(G, global_states)
    Ye = G.Y + ['delta']
    q0 = f"({G.x0},{G.y0})0"
    Q_new = [q0]
    Q = []
    Delta = []
    secret_logical_state = []

    while Q_new:
        q = Q_new[0]
        x, y, j = parse_state(q)
        x_g = f"({x},{y})"

        # Check if state is in secret time interval
        if x_g in G.SecretDwellTime:
            intervals = G.SecretDwellTime[x_g]
            current_time = int(j)
            for interval in intervals:
                if interval[0] <= current_time and (interval[1] == float('inf') or current_time < interval[1]):
                    secret_logical_state.append(q)
                    break

        # Process time transitions
        if int(j) < max_mapping[x_g]:
            next_j = str(int(j) + 1)
            q_bar = f"{x_g}{next_j}"
            Delta.append((q, 'delta', q_bar))
            if q_bar not in Q + Q_new:
                Q_new.append(q_bar)
        elif int(j) == max_mapping[x_g]:
            Delta.append((q, 'delta', q))

        # Process state transitions
        if int(j) > 0:
            for x_bar in sigma(x, G.B):
                if int(y) in G.h[x_bar]:
                    q_bar = f"({x_bar},{y})0"
                    Delta.append((q, 'epsilon', q_bar))
                    if q_bar not in Q + Q_new:
                        Q_new.append(q_bar)

                for y_bar in set(G.h[x_bar]) - {int(y)}:
                    q_bar = f"({x_bar},{y_bar})0"
                    Delta.append((q, str(y_bar), q_bar))
                    if q_bar not in Q + Q_new:
                        Q_new.append(q_bar)

            for y_bar in set(G.h[x]) - {int(y)}:
                q_bar = f"({x},{y_bar})0"
                Delta.append((q, str(y_bar), q_bar))
                if q_bar not in Q + Q_new:
                    Q_new.append(q_bar)

        Q_new.pop(0)
        Q.append(q)

    return EvolutionAutomaton(Q=Q, Ye=Ye, Delta=Delta, q0=q0, sls=secret_logical_state)

def construct_observer(G: SOA, Ge: EvolutionAutomaton) -> Observer:
    """Construct the observer automaton."""
    z0 = D_epsilon(Ge.q0, Ge)
    Z = [z0]
    Z_prime = []
    Delta_o = []

    while Z:
        z = Z[0]

        for y in Ge.Ye:
            alpha = []
            for q in z:
                xx = D_y(q, y, Ge)
                alpha.extend([x for x in xx if x not in alpha])

            beta = []
            for qq in alpha:
                xxx = D_epsilon(qq, Ge)
                beta.extend([x for x in xxx if x not in beta])

            z_bar = beta

            if z_bar and sorted(z_bar) not in [sorted(x) for x in Z + Z_prime]:
                Z.append(z_bar)

            if z_bar:
                transition = f"{','.join(sorted(z))} {y} {','.join(sorted(z_bar))}"
                if transition not in Delta_o:
                    Delta_o.append(transition)

        Z.pop(0)
        Z_prime.append(z)

    return Observer(Z=Z + Z_prime, Ye=Ge.Ye, Delta_o=Delta_o, z0=z0)

def verify_CSO(Ge: EvolutionAutomaton, Gobs: Observer) -> Tuple[bool, List[str]]:
    """Verify Current State Opacity."""
    CSO = True
    unopaque_array = []

    for current_state in Gobs.Z:
        current_states = set(current_state)
        secret_states = set(Ge.sls)

        if current_states.issubset(secret_states):
            CSO = False
            unopaque_array.append(','.join(sorted(current_state)))

    return CSO, unopaque_array

def parse_observer_transition(transition_str: str) -> Tuple[str, str, str]:
    """Parse observer transition string into from_state, event, and to_state."""
    parts = transition_str.split(' ')
    if len(parts) == 3:
        return parts[0], parts[1], parts[2]
    else:
        raise ValueError(f"Invalid transition format: {transition_str}")

def export_results_to_excel(Ge: EvolutionAutomaton, Gobs: Observer, filename: str):
    """Export results to Excel file."""
    # Create DataFrames for each component
    states_df = pd.DataFrame(Ge.Q, columns=['States'])
    transitions_df = pd.DataFrame(Ge.Delta, columns=['From_State', 'Event', 'To_State'])
    observer_states_df = pd.DataFrame([','.join(z) for z in Gobs.Z], columns=['Observer_States'])

    # Parse observer transitions into three columns
    from_states = []
    events = []
    to_states = []

    for transition in Gobs.Delta_o:
        from_state, event, to_state = parse_observer_transition(transition)
        from_states.append(from_state)
        events.append(event)
        to_states.append(to_state)

    observer_transitions_df = pd.DataFrame({
        'From_State': from_states,
        'Event': events,
        'To_State': to_states
    })

    # Get verification results
    CSO, unopaque_array = verify_CSO(Ge, Gobs)

    # Create verification results DataFrame
    verification_df = pd.DataFrame({
        'CSO_Result': [CSO],
        'Unopaque_States': [', '.join(unopaque_array) if unopaque_array else 'None']
    })

    # Export to Excel
    with pd.ExcelWriter(filename) as writer:
        states_df.to_excel(writer, sheet_name='Ge_States', index=False)
        transitions_df.to_excel(writer, sheet_name='Ge_Transitions', index=False)
        observer_states_df.to_excel(writer, sheet_name='Gobs_States', index=False)
        observer_transitions_df.to_excel(writer, sheet_name='Gobs_Transitions', index=False)
        verification_df.to_excel(writer, sheet_name='Verification_Results', index=False)

def simulate_soa_system(G: SOA, delta: float = 0.5, t_max: float = 10.0) -> Tuple[List[str], List[int]]:
    """Simulate the SOA system evolution."""
    t = np.arange(0, t_max + delta, delta)
    x = [G.x0]  # State trajectory
    y = [G.y0]  # Output trajectory

    for _ in t[1:]:
        # Find possible next states
        next_states = [b[1] for b in G.B if b[0] == x[-1]]
        # Randomly choose next state
        next_state = random.choice(next_states)
        x.append(next_state)
        # Randomly choose output from possible outputs
        y.append(random.choice(G.h[next_state]))

    return x, y

if __name__ == "__main__":
    # Define SOA system
    soa = SOA(
        X=['x0', 'x1', 'x2', 'x3'],
        Y=[1, 2, 3],
        h={'x0': [1, 2], 'x1': [2], 'x2': [1, 2], 'x3': [1, 3]},
        B=[('x0', 'x1'), ('x0', 'x2'), ('x1', 'x3'), ('x2', 'x3'), ('x3', 'x2')],
        x0='x0',
        y0=1,
        ss={'x2': [2]},
        SecretDwellTime={'(x2,2)': [[2, 4]]}
    )

    # Construct evolution automaton and observer
    Ge = construct_evolution_automaton(soa)
    Gobs = construct_observer(soa, Ge)

    # Export results
    export_results_to_excel(Ge, Gobs, 'SOA_results.xlsx')

    # Simulate system
    x, y = simulate_soa_system(soa)

    # Plot results
    t = np.arange(0, 10.5, 0.5)
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.step(t, [soa.X.index(state) for state in x], 'o-')
    plt.xlabel('Time')
    plt.ylabel('State')
    plt.yticks(range(len(soa.X)), soa.X)
    plt.title('State of the SOA system')
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.step(t, y, 'o-')
    plt.xlabel('Time')
    plt.ylabel('Output')
    plt.yticks(sorted(list(set([y for outputs in soa.h.values() for y in outputs]))))
    plt.title('Output of the SOA system')
    plt.grid(True)

    plt.tight_layout()
    plt.show()