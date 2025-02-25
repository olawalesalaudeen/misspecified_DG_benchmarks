import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import random

random.seed(1234)
np.random.seed(1234)

def generate_distribution(transformation_matrix=np.eye(n_features)):
    """
    Generate a synthetic dataset with causal and environmental features.

    Parameters:
    - transformation_matrix: numpy.ndarray, default=identity matrix
        Transformation matrix for environmental features

    Returns:
    - dict with keys:
        'X': Combined features (causal + environmental)
        'Y': Labels (-1 or 1)
        'Zc': Causal features
        'Ze': Environmental features
    """
    true_labels = np.random.choice([-1, 1], p=[1-p_y1, p_y1], size=n_samples)
    label_noise = np.random.choice([-1, 1], p=[p_noise, 1-p_noise], size=n_samples)
    noisy_labels = true_labels * label_noise

    causal_features = true_labels.reshape(-1,1) * np.random.multivariate_normal(mu_c, Sigma_c, n_samples)
    env_features = noisy_labels.reshape(-1,1) * np.random.multivariate_normal(
        transformation_matrix@mu_e,
        transformation_matrix@Sigma_e@transformation_matrix.T,
        n_samples
    )
    combined_features = np.hstack((causal_features, env_features))

    return {
        'X': combined_features,
        'Y': noisy_labels,
        'Zc': causal_features,
        'Ze': env_features
    }

def generate_mixture_distribution(transformation_matrices, mixture_weights=None):
    """
    Generate a mixture distribution based on provided transformation matrices and probabilities.

    Parameters:
    - transformation_matrices: list of numpy.ndarray
        List of transformation matrices for each mixture component.
    - mixture_weights: list of float
        List of probabilities for each mixture component (must sum to 1).

    Returns:
    - dict
        A dictionary containing generated data ('X', 'Y', 'Zc', 'Ze').

    """
    if mixture_weights is None:
        mixture_weights = [1 / len(transformation_matrices)] * len(transformation_matrices)

    if not np.isclose(sum(mixture_weights), 1):
        raise ValueError("Mixture weights must sum to 1.")

    # Allocate samples per component
    component_sizes = [int(weight * n_samples) for weight in mixture_weights]
    component_sizes[-1] = n_samples - sum(component_sizes[:-1])  # Adjust last to ensure exact n_samples

    all_features = []
    all_labels = []
    all_causal = []
    all_env = []

    for matrix, size in zip(transformation_matrices, component_sizes):
        generated_data = generate_distribution(matrix)
        all_features.append(generated_data['X'])
        all_labels.append(generated_data['Y'])
        all_causal.append(generated_data['Zc'])
        all_env.append(generated_data['Ze'])

    # Combine all components
    combined_features = np.vstack(all_features)
    combined_labels = np.concatenate(all_labels)
    combined_causal = np.vstack(all_causal)
    combined_env = np.vstack(all_env)

    return {
        'X': combined_features,
        'Y': combined_labels,
        'Zc': combined_causal,
        'Ze': combined_env
    }

def train_and_test_model(train_features, train_labels, test_datasets):
    """
    Train a logistic regression model and evaluate it on multiple test sets.

    Parameters:
    - train_features: numpy.ndarray
        Training features
    - train_labels: numpy.ndarray
        Training labels
    - test_datasets: list of tuples
        List of (test_features, test_labels) pairs to evaluate on

    Returns:
    - tuple:
        - trained LogisticRegression model
        - list of accuracy scores for each test set
    """
    model = LogisticRegression(random_state=0).fit(train_features, train_labels)
    accuracy_scores = [
        accuracy_score(test_labels, model.predict(test_features))
        for test_features, test_labels in test_datasets
    ]
    return model, accuracy_scores

def generate_Ms(n, d, bound=10, close_to_identity=False, include_nd=True):
    """
    Generate a list of positive definite and optionally negative definite matrices.

    Parameters:
    - n: int
        Size of the matrices (n x n)
    - d: int
        Total number of matrices to generate
    - bound: float, default=10
        Maximum absolute value for the matrix norm
    - close_to_identity: bool, default=False
        If True, generates matrices closer to the identity matrix
    - include_nd: bool, default=True
        If True, includes negative definite matrices in the output

    Returns:
    - list of numpy.ndarray
        List of matrices where each matrix has norm <= bound. If include_nd is True,
        returns alternating positive and negative definite matrices.
    """
    Ms = []
    E = max(1, d // 2) if include_nd else d
    for _ in range(E):
        M = generate_pd_nd_pair(n, bound, close_to_identity=close_to_identity)
        if include_nd:
            Ms += M
        else:
            Ms += M[0:1]
    return Ms

def generate_pd_nd_pair(n, bound=10, close_to_identity=False):
    """
    Generate a pair of positive definite (PD) and negative definite (ND) matrices.

    The matrices are generated by creating a random symmetric matrix and manipulating
    its eigenvalues to ensure positive/negative definiteness. The matrices are then
    scaled to have a Frobenius norm within the specified bound.

    Parameters:
    - n: int
        Size of the matrices (n x n)
    - bound: float, default=10
        Maximum absolute value for the matrix norm
    - close_to_identity: bool, default=False
        If True, generates matrices closer to the identity matrix

    Returns:
    - list of [pd_matrix, nd_matrix]:
        pd_matrix: Positive definite matrix with norm <= bound
        nd_matrix: Negative definite matrix with norm <= bound
    """
    # Generate a random symmetric matrix
    A = np.random.randn(n, n)
    symmetric_matrix = (A + A.T) / 2  # Make it symmetric

    if close_to_identity:
        symmetric_matrix = symmetric_matrix * 0. + np.eye(n)  # Blend with identity matrix

    # Positive definite matrix
    eig_values = np.abs(np.linalg.eigvals(symmetric_matrix)) + 0.1  # Ensure positive eigenvalues
    pd_matrix = symmetric_matrix @ np.diag(eig_values) @ symmetric_matrix.T

    # Normalize the PD matrix to have a norm of 1
    size = np.random.rand() * bound
    norm_pd = np.linalg.norm(pd_matrix, ord='fro')  # Frobenius norm
    pd_matrix = (pd_matrix / norm_pd) * size # Scale to the desired bound

    # Negative definite matrix
    nd_matrix = -pd_matrix  # Negate to make eigenvalues negative

    return [pd_matrix, nd_matrix]

def get_inner_product(transformation_matrix, model):
    """
    Compute the inner product w^T(M * mu_e) and its variance.

    This function calculates both the mean prediction and its variance under the
    environmental transformation M. The variance is computed as sqrt(2 * w^T M Sigma_e M^T w).

    Parameters:
    - transformation_matrix: numpy.ndarray
        Transformation matrix for environmental features
    - model: sklearn.linear_model.LogisticRegression
        Trained logistic regression classifier

    Returns:
    - tuple:
        - numpy.ndarray: Inner product w^T(M * mu_e)
        - numpy.ndarray: Square root of variance term

    Raises:
    - ValueError: If matrix dimensions don't match with mu_e or Sigma_e
    """
    global mu_e, Sigma_e  # Access the global variables

    # Validate dimensions
    if transformation_matrix.shape[1] != mu_e.shape[0]:
        raise ValueError("mu_e size must match the number of features in transformation matrix.")
    if Sigma_e.shape[0] != transformation_matrix.shape[1] or Sigma_e.shape[1] != transformation_matrix.shape[1]:
        raise ValueError("Sigma_e dimensions must match transformation matrix dimensions.")

    # Get model coefficients for environmental features
    env_coefficients = model.coef_[:, n_features:].flatten()
    model_intercept = model.intercept_[0]

    # Calculate transformed mean
    transformed_mean = transformation_matrix @ mu_e
    mean_prediction = env_coefficients.T @ transformed_mean

    # Calculate prediction variance
    variance_sqrt = np.sqrt(2 * env_coefficients.T @ transformation_matrix @ Sigma_e @ transformation_matrix.T @ env_coefficients)

    return mean_prediction, variance_sqrt
