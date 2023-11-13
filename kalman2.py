import numpy as np
import matplotlib.pyplot as plt

# True state
num_steps = 10
true_xs = np.linspace(0, 10, num=num_steps + 1) 
true_ys = np.linspace(0, 10, num=num_steps + 1)  
true_states = np.stack((true_xs,true_ys), axis=1) 

# Linear stochastic difference
x_0, y_0 = 0, 0 
mo_states = [np.array([x_0, y_0])] 
u_t = np.array([1.0, 1.0]) 
A = np.array([[1, 0],
              [0, 1]])
B = np.array([[1, 0],
              [0, 1]])
Q = np.array([[1, 0],
              [0, 1]]) 
for _ in range(10):
    mo_noise = np.random.multivariate_normal(mean=np.array([0,0]), cov=Q) 
    new_state = A @ mo_states[-1] + B @ u_t + mo_noise 
    mo_states.append(new_state)

# Measurements 
me_states = [np.array([x_0, y_0])] 
H = np.array([[1, 0],
              [0, 1]]) 
R = np.array([[0.5, 0],
              [0, 0.5]]) 
for i in range(10):
    me_noise = np.random.multivariate_normal(mean=np.array([0,0]), cov=R) 
    new_me = H @ true_states[i+1] + me_noise 
    me_states.append(new_me)

mo_states = np.array(mo_states)
me_states = np.array(me_states)

# Init
mu_0 = np.array([0, 0])
sig_0 = np.array([[0.1, 0],
                     [0, 0.1]])
u_t = np.array([1, 1]) 

me_states = []
fil_states = []

def predict(A, B, Q, u_k, mu_k, sig_k):
    pred_mu = A @ mu_k + B @ u_k
    pred_sig = A @ sig_k @ A.T + Q
    return pred_mu, pred_sig

def update(H, R, z, pred_mu, pred_sig):
    resi_mean = z - H @ pred_mu
    resi_cov = H @ pred_sig @ H.T + R
    kalman_gain = pred_sig @ H.T @ np.linalg.inv(resi_cov)
    upd_mu = pred_mu + kalman_gain @ resi_mean
    upd_sig = pred_sig - kalman_gain @ H @ pred_sig
    return upd_mu, upd_sig

# Kalman filter
mu_cur = mu_0.copy()
sig_cur = sig_0.copy()
for i in range(num_steps):
    # Predict
    pred_mu, pred_sig = predict(A, B, Q, u_t, mu_cur, sig_cur)
    # Measurement 
    me_noise = np.random.multivariate_normal(mean=np.array([0,0]), cov=R) 
    new_me = H @ true_states[i+1] + me_noise 
    # Update
    mu_cur, sig_cur = update(H, R, new_me, pred_mu, pred_sig)
    # Store
    me_states.append(new_me)
    fil_states.append(mu_cur)

me_states = np.array(me_states)
fil_states = np.array(fil_states)

# Results


plt.plot(true_states[:,0], true_states[:,1]) 
plt.plot(mo_states[:,0], mo_states[:,1]) 
plt.plot(me_states[:,0], me_states[:,1])
plt.plot(fil_states[:,0], fil_states[:,1])
plt.plot(me_states,'k+',label='Measurements')
plt.xlim(-1,12)
plt.ylim(-1,12)
plt.xlabel('x position')
plt.ylabel('y position')
plt.legend(['True state', 'Linear stochastic difference', 'Measurements', 'Kalman Filter'])
plt.gca().set_aspect('equal', adjustable='box')
plt.show()