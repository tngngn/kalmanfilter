import numpy as np
import matplotlib.pyplot as plt

num_steps = 50

#True state
true_x = -0.37727 

# Measurements 
z = np.random.normal(true_x,0.1,size=(num_steps,)) 

Q = 1e-5 
R = 0.1**2 

# Init
mu_k=np.zeros((num_steps,))      
Sigma_k=np.zeros((num_steps,))         
predicted_mu=np.zeros((num_steps,))
predicted_Sigma=np.zeros((num_steps,))   
K=np.zeros((num_steps,))        
mu_k[0] = 0.0
Sigma_k[0] = 1.0

# Kalman filter
for k in range(1,num_steps):
    # Time update
    predicted_mu[k] = mu_k[k-1]
    predicted_Sigma[k] = Sigma_k[k-1]+Q

    # Measurement update
    K[k] = predicted_Sigma[k]/( predicted_Sigma[k]+R )
    mu_k[k] = predicted_mu[k]+K[k]*(z[k]-predicted_mu[k])
    Sigma_k[k] = (1-K[k])*predicted_Sigma[k]

plt.figure()
plt.plot(z,'k+',label='Measurements')
plt.plot(mu_k,'b-',label='Kalman Filter')
plt.axhline(true_x,color='g',label='True State')
plt.legend()
plt.title('Estimate vs. iteration step', fontweight='bold')
plt.xlabel('Iteration')
plt.ylabel('Voltage')

plt.figure()
valid_iter = range(1,num_steps) 
plt.plot(valid_iter,predicted_Sigma[valid_iter],label='a priori error estimate')
plt.title('Estimated $\it{\mathbf{a \ priori}}$ error vs. iteration step', fontweight='bold')
plt.xlabel('Iteration')
plt.ylabel('$(Voltage)^2$')
plt.setp(plt.gca(),'ylim',[0,.01])
plt.show()