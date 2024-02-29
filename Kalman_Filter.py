# -*- coding: utf-8 -*-
"""
Created on Thu Feb 29 00:42:48 2024

@author: jayni
"""

class KalmanFilter:
    """
    Implements a basic Kalman Filter with customizable system dynamics.
    
    Attributes:
        initial_state (np.matrix): Initial state estimate.
        initial_error_covariance (np.matrix): Initial error covariance.
        transition_matrix (np.matrix): State transition matrix.
        control_matrix (np.matrix): Control matrix.
        observation_matrix (np.matrix): Observation matrix.
        process_noise_covariance (np.matrix): Process noise covariance.
        measurement_noise_covariance (np.matrix): Measurement noise covariance.
        time_step (int): Current time step of the filter.
        posteriori_estimates (list): List of posteriori state estimates.
        priori_estimates (list): List of priori state estimates.
        posteriori_error_covariances (list): List of posteriori error covariance matrices.
        priori_error_covariances (list): List of priori error covariance matrices.
        kalman_gains (list): List of Kalman gain matrices.
        measurement_residuals (list): List of measurement residuals.
    """
    
    def __init__(self, initial_state, initial_error_covariance, transition_matrix, control_matrix, observation_matrix, process_noise_covariance, measurement_noise_covariance):
        """
        Initializes the Kalman Filter with the specified system matrices and initial conditions.
        """
        self.initial_state = initial_state
        self.initial_error_covariance = initial_error_covariance
        self.transition_matrix = transition_matrix
        self.control_matrix = control_matrix
        self.observation_matrix = observation_matrix
        self.process_noise_covariance = process_noise_covariance
        self.measurement_noise_covariance = measurement_noise_covariance
        
        self.time_step = 0
        
        self.posteriori_estimates = [initial_state]
        self.priori_estimates = []
        
        self.posteriori_error_covariances = [initial_error_covariance]
        self.priori_error_covariances = []
        
        self.kalman_gains = []
        self.measurement_residuals = []
    
    def propagate_dynamics(self, control_input):
        """
        Propagates the state and error covariance to the next time step.
        """
        last_estimate = self.posteriori_estimates[self.time_step]
        last_error_covariance = self.posteriori_error_covariances[self.time_step]
        
        priori_estimate = self.transition_matrix * last_estimate + self.control_matrix * control_input
        priori_error_covariance = self.transition_matrix * last_error_covariance * self.transition_matrix.T + self.process_noise_covariance
        
        self.priori_estimates.append(priori_estimate)
        self.priori_error_covariances.append(priori_error_covariance)
        
        self.time_step += 1
    
    def update(self, measurement):
        """
        Updates the state estimate and error covariance based on the given measurement.
        """
        import numpy as np
        
        K = self.priori_error_covariances[self.time_step - 1] * self.observation_matrix.T * np.linalg.inv(self.measurement_noise_covariance + self.observation_matrix * self.priori_error_covariances[self.time_step - 1] * self.observation_matrix.T)
        
        measurement_residual = measurement - self.observation_matrix * self.priori_estimates[self.time_step - 1]
        
        posteriori_estimate = self.priori_estimates[self.time_step - 1] + K * measurement_residual
        
        identity_matrix = np.eye(self.initial_state.shape[0])
        posteriori_error_covariance = (identity_matrix - K * self.observation_matrix) * self.priori_error_covariances[self.time_step - 1] * (identity_matrix - K * self.observation_matrix).T + K * self.measurement_noise_covariance * K.T
        
        self.kalman_gains.append(K)
        self.measurement_residuals.append(measurement_residual)
        self.posteriori_estimates.append(posteriori_estimate)
        self.posteriori_error_covariances.append(posteriori_error_covariance)
