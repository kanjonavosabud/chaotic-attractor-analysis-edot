import numpy as np
import networkx as nx
from math import comb

def create_reservoir(output_dim=3, reservoir_dim=300, spectral_radius=1.2, edge_prob=.1, seed=None, 
                    graph_type="default", random_edge_prob=False, beta=1.0):

    # Initialize components
    reservoir = np.zeros(reservoir_dim)
    reservoir_input = (np.random.rand(reservoir_dim, output_dim) - .5) * edge_prob
    
    # Initialize constant vector C with random values from -beta to beta
    C = np.random.uniform(-beta, beta, reservoir_dim)
    
    # Create graph based on type
    if graph_type == "default":
        graph = _create_default_graph(reservoir_dim, edge_prob, seed, random_edge_prob)
        adj_matrix = nx.to_numpy_array(graph)  # Create adjacency matrix for default graph
    elif graph_type == "complete":
        graph = nx.complete_graph(reservoir_dim)
        adj_matrix = nx.to_numpy_array(graph)  # Create adjacency matrix for complete graph
    elif graph_type == "tree":
        graph = nx.random_powerlaw_tree(reservoir_dim, seed=seed, tries=reservoir_dim*1000)
        adj_matrix = nx.to_numpy_array(graph)  # Create adjacency matrix for tree graph
    elif graph_type == "zero":
        # Create a zero matrix with the same dimensions as the adjacency matrix
        adj_matrix = np.zeros((reservoir_dim, reservoir_dim))  # Same dimensions as the other graphs
        return adj_matrix, reservoir, reservoir_input, C

    
    # Convert to numpy array and scale
    rand_mat = 2 * (np.random.rand(reservoir_dim, reservoir_dim) - .5)
    scale_mat = adj_matrix * rand_mat
    
    # Scale by spectral radius
    max_eigenval = np.max(np.linalg.eig(scale_mat)[0])
    rescaled_mat = scale_mat / np.abs(max_eigenval) * spectral_radius
    
    return rescaled_mat, reservoir, reservoir_input, C

def fit_reservoir(data, reservoir_dim, reservoir, adj_matrix, reservoir_input, C):

    output_dim = data.shape[1]
    state_matrix = np.zeros((reservoir_dim, len(data)))
    
    # Compute reservoir states
    for t in range(len(data)):
        state_matrix[:, t] = reservoir
        x = np.dot(adj_matrix, reservoir) + np.dot(reservoir_input, data[t]) + C
        reservoir = np.tanh(x)
    
    # Compute output weights
    state_T = state_matrix.T
    reg_factor = 0.0001
    identity = reg_factor * np.eye(reservoir_dim)
    inverse = np.linalg.solve(np.dot(state_matrix, state_T) + identity, np.eye(reservoir_dim))
    reservoir_output = np.dot(np.dot(data.T, state_T), inverse)
    
    return reservoir, reservoir_output

def predict(steps, output_dim, reservoir, adj_matrix, reservoir_input, reservoir_output, C):

    predictions = np.zeros((steps, output_dim))
    for t in range(steps):
        predictions[t] = np.dot(reservoir_output, reservoir)
        x = np.dot(adj_matrix, reservoir) + np.dot(reservoir_input, predictions[t]) + C
        reservoir = np.tanh(x)
    return predictions 

def _create_default_graph(reservoir_dim, edge_prob, seed=None, random_edge_prob=False):

    if not random_edge_prob:
        # Simple random graph with fixed edge probability
        return nx.gnp_random_graph(reservoir_dim, edge_prob, seed)
    
    # Random graph with variable edge probabilities
    graph = nx.Graph()
    for i in range(reservoir_dim):
        graph.add_node(i)
    
    # Calculate expected number of edges
    expected_edges = int(comb(reservoir_dim, 2) * 0.5)
    edges_added = 0
    
    while edges_added < expected_edges:
        # Select random nodes
        node1 = np.random.randint(0, reservoir_dim)
        node2 = np.random.randint(0, reservoir_dim)
        
        # Ensure different nodes
        if node1 == node2:
            continue
            
        # Add edge with random probability if not already present
        if node1 not in graph[node2]:
            probability = np.random.uniform(0, 1)
            if np.random.random() < probability:
                graph.add_edge(node1, node2)
                graph.add_edge(node2, node1)
                edges_added += 1
    
    return graph 

def fit_reservoir_case1(data, reservoir_dim, reservoir, adj_matrix, reservoir_input, C):
    """
    Fit output weights using one-step-ahead prediction loss (Case 1).
    """
    output_dim = data.shape[1]
    state_matrix = np.zeros((reservoir_dim, len(data)-1))
    targets = np.zeros((output_dim, len(data)-1))
    for t in range(len(data)-1):
        state_matrix[:, t] = reservoir
        x = np.dot(adj_matrix, reservoir) + np.dot(reservoir_input, data[t]) + C
        reservoir = np.tanh(x)
        targets[:, t] = data[t+1]  # one-step-ahead target
    
    # Number of samples
    N = len(data) - 1
    
    # Scale state_matrix and targets by 1/sqrt(N)
    # state_matrix /= np.sqrt(N)
    # targets /= np.sqrt(N)
    state_matrix /= N
    targets /= N

    # Ridge regression
    state_T = state_matrix.T
    reg_factor = 0.0001
    identity = reg_factor * np.eye(reservoir_dim)
    inverse = np.linalg.solve(np.dot(state_matrix, state_T) + identity, np.eye(reservoir_dim))
    reservoir_output = np.dot(np.dot(targets, state_T), inverse)
    return reservoir, reservoir_output


def fit_reservoir_case2(data, reservoir_dim, reservoir, adj_matrix, reservoir_input, C, beta, rho):
    """
    Fit output weights using fixed point + one-step-ahead prediction loss (Case 2).
    Fixed points are computed internally from beta and rho.
    """
    output_dim = data.shape[1]
    state_matrix = np.zeros((reservoir_dim, len(data)-1))
    targets = np.zeros((output_dim, len(data)-1))
    for t in range(len(data)-1):
        state_matrix[:, t] = reservoir
        x = np.dot(adj_matrix, reservoir) + np.dot(reservoir_input, data[t]) + C
        reservoir = np.tanh(x)
        targets[:, t] = data[t+1]  # one-step-ahead target
    
    # Number of samples
    N = len(data) - 1
    
    # Scale state_matrix and targets by 1/sqrt(N)
    # state_matrix /= np.sqrt(N) 
    # targets /= np.sqrt(N) 
    state_matrix /= N
    targets /= N

    # Compute fixed points
    fp1 = np.array([np.sqrt(beta * (rho - 1)), np.sqrt(beta * (rho - 1)), rho - 1])
    fp2 = np.array([-np.sqrt(beta * (rho - 1)), -np.sqrt(beta * (rho - 1)), rho - 1])
    fixed_points = [fp1, fp2]
    
    # Fixed point terms
    r_fixed = []
    for u_fp in fixed_points:
        r = np.tanh(np.dot(reservoir_input, u_fp) + C)  # memoryless
        r_fixed.append(r)
    r_fixed = np.stack(r_fixed, axis=1)  # shape: (reservoir_dim, 2)
    u_fixed = np.stack(fixed_points, axis=1)  # shape: (3, 2)
    
    # Combine state matrix and fixed point states
    state_matrix_full = np.concatenate([state_matrix, r_fixed], axis=1)
    targets_full = np.concatenate([targets, u_fixed], axis=1)
    
    # Ridge regression
    state_T = state_matrix_full.T
    reg_factor = 0.0001
    identity = reg_factor * np.eye(reservoir_dim)
    inverse = np.linalg.solve(np.dot(state_matrix_full, state_T) + identity, np.eye(reservoir_dim))
    reservoir_output = np.dot(np.dot(targets_full, state_T), inverse)
    return reservoir, reservoir_output

def fit_reservoir_case3(data, reservoir_dim, reservoir, adj_matrix, reservoir_input, C, beta, rho):
    """
    Fit output weights using a combination of on-attractor, fixed point, and off-attractor dynamics (Case 3).
    """
    output_dim = data.shape[1]
    state_matrix = np.zeros((reservoir_dim, len(data)-1))
    targets = np.zeros((output_dim, len(data)-1))
    for t in range(len(data)-1):
        state_matrix[:, t] = reservoir
        x = np.dot(adj_matrix, reservoir) + np.dot(reservoir_input, data[t]) + C
        reservoir = np.tanh(x)
        targets[:, t] = data[t+1]  # one-step-ahead target
    
    # Number of samples
    N = len(data) - 1
    
    # Scale state_matrix and targets by 1/sqrt(N)
    state_matrix /= np.sqrt(N)
    targets /= np.sqrt(N)

    # Compute fixed points
    fp1 = np.array([np.sqrt(beta * (rho - 1)), np.sqrt(beta * (rho - 1)), rho - 1])
    fp2 = np.array([-np.sqrt(beta * (rho - 1)), -np.sqrt(beta * (rho - 1)), rho - 1])
    fixed_points = [fp1, fp2]
    
    # Fixed point terms
    r_fixed = []
    for u_fp in fixed_points:
        r = np.tanh(np.dot(reservoir_input, u_fp) + C)  # memoryless
        r_fixed.append(r)
    r_fixed = np.stack(r_fixed, axis=1)  # shape: (reservoir_dim, 2)
    u_fixed = np.stack(fixed_points, axis=1)  # shape: (3, 2)
    
    # Off-attractor dynamics
    off_attractor_matrix = np.zeros((reservoir_dim, 5 * len(data)))
    off_targets = np.zeros((output_dim, 5 * len(data)))
    for i, init_point in enumerate(data):
        for j in range(5):
            orbit_point = init_point + (j + 1) * np.random.normal(0, 0.1, size=output_dim)  # Simulate orbit
            x = np.dot(adj_matrix, reservoir) + np.dot(reservoir_input, orbit_point) + C
            reservoir = np.tanh(x)
            off_attractor_matrix[:, 5 * i + j] = reservoir
            off_targets[:, 5 * i + j] = orbit_point
    
    # Normalize off-attractor terms
    off_attractor_matrix /= np.sqrt(5 * len(data))
    off_targets /= np.sqrt(5 * len(data))

    # Combine all terms
    state_matrix_full = np.concatenate([state_matrix, r_fixed, off_attractor_matrix], axis=1)
    targets_full = np.concatenate([targets, u_fixed, off_targets], axis=1)
    
    # Ridge regression
    state_T = state_matrix_full.T
    reg_factor = 0.0001
    identity = reg_factor * np.eye(reservoir_dim)
    inverse = np.linalg.solve(np.dot(state_matrix_full, state_T) + identity, np.eye(reservoir_dim))
    reservoir_output = np.dot(np.dot(targets_full, state_T), inverse)
    return reservoir, reservoir_output

def run_reservoir(train_data, test_data, adj_matrix=None, reservoir_params=None, fit_mode='default'):
    """
    Create, train and get predictions from a reservoir
    fit_mode: 'default' (identity), 'case1' (one-step), 'case2' (fixed points + one-step)
    """
    if reservoir_params is None:
        reservoir_params = {
            'output_dim': 3,
            'reservoir_dim': 300,
            'spectral_radius': 1.2,
            'edge_prob': 0.1,
            'graph_type': 'zero',
            'beta': 1.0
        }
    if adj_matrix is None:
        adj_matrix, reservoir, res_input, C = create_reservoir(**reservoir_params)
    else:
        _, reservoir, res_input, C = create_reservoir(**reservoir_params)
    # Train reservoir
    if fit_mode == 'case1':
        reservoir, res_output = fit_reservoir_case1(
            train_data,
            reservoir_params['reservoir_dim'],
            reservoir,
            adj_matrix,
            res_input,
            C
        )
    elif fit_mode == 'case2':
        # Extract beta and rho from params or use defaults
        beta = reservoir_params.get('beta', 8/3)
        rho = reservoir_params.get('rho', 24)
        reservoir, res_output = fit_reservoir_case2(
            train_data,
            reservoir_params['reservoir_dim'],
            reservoir,
            adj_matrix,
            res_input,
            C,
            beta,
            rho
        )
    elif fit_mode == 'case3':
        beta = reservoir_params.get('beta', 8/3)
        rho = reservoir_params.get('rho', 24)
        reservoir, res_output = fit_reservoir_case3(
            train_data,
            reservoir_params['reservoir_dim'],
            reservoir,
            adj_matrix,
            res_input,
            C,
            beta,
            rho
        )
    else:
        reservoir, res_output = fit_reservoir(
            train_data,
            reservoir_params['reservoir_dim'],
            reservoir,
            adj_matrix,
            res_input,
            C
        )
    # Make predictions
    predictions = predict(
        len(test_data),
        reservoir_params['output_dim'],
        reservoir,
        adj_matrix,
        res_input,
        res_output,
        C
    )
    return {
        'predictions': predictions,
        'adj_matrix': adj_matrix,
        'reservoir': reservoir,
        'res_input': res_input,
        'res_output': res_output,
        'C': C
    }