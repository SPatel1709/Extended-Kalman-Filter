import numpy as np

from utils import minimized_angle


class ExtendedKalmanFilter:
    def __init__(self, mean, cov, alphas, beta):
        self.alphas = alphas
        self.beta = beta

        self._init_mean = mean
        self._init_cov = cov
        self.reset()

    def reset(self):
        self.mu = self._init_mean
        self.sigma = self._init_cov

    def update(self, env, u, z, marker_id):
        """ Update the state estimate after taking action and receiving a landmark
        observation.

        u: action
        z: landmark observation
        marker_id: landmark ID
        """
        # YOUR IMPLEMENTATION HERE

        G=env.G(self.mu,u)
        V=env.V(self.mu,u)

        #update the state variables
        mu_new =env.forward(self.mu,u)

        #jacobian for all the models state, control and observation model
       
        H=env.H(mu_new,marker_id)

        variances = np.zeros(3)
        variances[0] = env.alphas[1] *( u[0]**2) + env.alphas[0] * (u[1]**2)            
        variances[1] = env.alphas[3] * (u[1]**2) + env.alphas[1] * (u[0]**2 + u[2]**2)
        variances[2] = env.alphas[1] * (u[2]**2) + env.alphas[3] * (u[1]**2)
        Q=np.diag(variances)
        
        #covariance prediction
        #Temp=np.dot(np.dot(G,self.sigma),G.T)+np.dot(np.dot(V,Q),V.T)
        #Kalman gain=temp*H(transpose)*
        K=np.dot(np.dot(np.dot(np.dot(G,self.sigma),G.T)+np.dot(np.dot(V,Q),V.T),H.T),np.linalg.inv(np.dot(H,np.dot(np.dot(np.dot(G,self.sigma),G.T)+np.dot(np.dot(V,Q),V.T),H.T))+env.beta))

        
        theta_bearing=env.observe(mu_new,marker_id)

        #state update
        self.mu=mu_new + K*(minimized_angle(z[0]-(theta_bearing)))
        #covariance update
        self.sigma=np.dot(([[1,0,0],[0,1,0],[0,0,1]]-np.dot(K,H)),np.dot(np.dot(G,self.sigma),G.T)+np.dot(np.dot(V,Q),V.T))
        
        return self.mu, self.sigma
