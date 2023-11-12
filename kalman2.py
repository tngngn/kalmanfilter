import numpy as np
import matplotlib.pyplot as plt

# True state
num_steps = 10
true_xs = np.linspace(0, 10, num=num_steps + 1) 
true_ys = np.linspace(0, 10, num=num_steps + 1)  
true_states = np.stack((true_xs,true_ys), axis=1) 

# Linear stochastic difference
x_0, y_0 = 0, 0 
motion_states = [np.array([x_0, y_0])] 
u_t = np.array([1.0, 1.0]) 
A = np.array([[1, 0],
              [0, 1]])
B = np.array([[1, 0],
              [0, 1]])
Q = np.array([[0.65, 0],
              [0, 0.75]]) 
for _ in range(10):
    motion_noise = np.random.multivariate_normal(mean=np.array([0,0]), cov=Q) 
    new_state = A @ motion_states[-1] + B @ u_t + motion_noise 
    motion_states.append(new_state)

# Measurements 
measurement_states = [np.array([x_0, y_0])] 
H = np.array([[1, 0],
              [0, 1]]) 
R = np.array([[0.05, 0],
              [0, 0.02]]) 
for i in range(10):
    measurement_noise = np.random.multivariate_normal(mean=np.array([0,0]), cov=R) 
    new_measurement = H @ true_states[i+1] + measurement_noise 
    measurement_states.append(new_measurement)

motion_states = np.array(motion_states)
measurement_states = np.array(measurement_states)

# Init
mu_0 = np.array([0, 0])
Sigma_0 = np.array([[0.1, 0],
                     [0, 0.1]])
u_t = np.array([1, 1]) 
A = np.array([[1, 0],
              [0, 1]])
B = np.array([[1, 0],
              [0, 1]])
Q = np.array([[0.3, 0],
              [0, 0.3]])
H = np.array([[1, 0],
              [0, 1]])
R = np.array([[0.75, 0],
              [0, 0.6]])

measurement_states = []
filtered_states = []

def predict(A, B, Q, u_k, mu_k, Sigma_k):
    predicted_mu = A @ mu_k + B @ u_k
    predicted_Sigma = A @ Sigma_k @ A.T + Q
    return predicted_mu, predicted_Sigma

def update(H, R, z, predicted_mu, predicted_Sigma):
    residual_mean = z - H @ predicted_mu
    residual_covariance = H @ predicted_Sigma @ H.T + R
    kalman_gain = predicted_Sigma @ H.T @ np.linalg.inv(residual_covariance)
    updated_mu = predicted_mu + kalman_gain @ residual_mean
    updated_Sigma = predicted_Sigma - kalman_gain @ H @ predicted_Sigma
    return updated_mu, updated_Sigma

# Kalman filter
mu_current = mu_0.copy()
Sigma_current = Sigma_0.copy()
for i in range(num_steps):
    # Predict
    predicted_mu, predicted_Sigma = predict(A, B, Q, u_t, mu_current, Sigma_current)
    # Measurement 
    measurement_noise = np.random.multivariate_normal(mean=np.array([0,0]), cov=R) # ~N(0,R)
    new_measurement = H @ true_states[i+1] + measurement_noise # this is z_t
    # Update
    mu_current, Sigma_current = update(H, R, new_measurement, predicted_mu, predicted_Sigma)
    # Store
    measurement_states.append(new_measurement)
    filtered_states.append(mu_current)

measurement_states = np.array(measurement_states)
filtered_states = np.array(filtered_states)

# Results
plt.plot(true_states[:,0], true_states[:,1]) 
plt.plot(motion_states[:,0], motion_states[:,1]) 
plt.plot(measurement_states[:,0], measurement_states[:,1])
plt.plot(filtered_states[:,0], filtered_states[:,1])
plt.xlim(-1,12)
plt.ylim(-1,12)
plt.xlabel('x position')
plt.ylabel('y position')
plt.legend(['True state', 'Linear stochastic difference', 'Measurements', 'Kalman Filter'])
plt.gca().set_aspect('equal', adjustable='box')
plt.show()