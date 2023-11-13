import numpy as np
import matplotlib.pyplot as plt

num_steps = 50

#True state
true_x = -0.37727 

# Measurements 
z = np.random.normal(true_x,0.1,size=(num_steps,)) 

Q = 1e-5 
R = 0.1**4

# Init
mu_k=np.zeros((num_steps,))      
sig_k=np.zeros((num_steps,))         
pred_mu=np.zeros((num_steps,))
pred_sig=np.zeros((num_steps,))   
K=np.zeros((num_steps,))        
mu_k[0] = 0.0
sig_k[0] = 1.0

# Kalman filter
for k in range(1,num_steps):
    # Time update
    pred_mu[k] = mu_k[k-1]
    pred_sig[k] = sig_k[k-1]+Q

    # Measurement update
    kalman_gain[k] = pred_sig[k]/( pred_sig[k]+R )
    mu_k[k] = pred_mu[k]+kalman_gain[k]*(z[k]-pred_mu[k])
    sig_k[k] = (1-kalman_gain[k])*pred_sig[k]

plt.figure()
plt.plot(z,'k+',label='Measurements')
plt.plot(mu_k,'b-',label='Kalman Filter')
plt.axhline(true_x,color='g',label='True State')
plt.legend()
plt.title('Estimate vs. iteration step', fontweight='bold')
plt.xlabel('Iteration')
plt.ylabel('Voltage')
