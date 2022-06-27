
from math import factorial
import numpy as np
import matplotlib.pyplot as plt
import manifpy as mf

"""
Vehicle Model
"""

class DifferentialDrive:
        def __init__(self, wheel_distance):
                self._wheel_distance = wheel_distance

        def forward(self, v, state, dt):
                vl,vr = v
                l = self._wheel_distance
                r = l/2 * (vl + vr)
                omega = (vr - vl)/l
                x,y,theta = state
                icc = np.array([x-r*np.sin(theta), y + r*np.cos(theta)])
                R = np.array([
                        [np.cos(omega*dt), -np.sin(omega*dt), 0],
                        [np.sin(omega*dt), np.cos(omega*dt), 0],
                        [0, 0, 1],
                ])
                p = np.array([x-icc[0], y-icc[1], theta]).transpose()

                t = np.array([icc[0],icc[1],omega*dt]).transpose()
                #print(f"R={R}\np={p},t={t}")
                return R.dot(p) + t

class SE2:
        DoF = 3

        def __init__(self,T):
                self._T = T

        def __mul__(self,that):
                if type(that) is SE2:
                        return SE2(self._T @ that._T)
                if type(that) is np.array:
                        assert that.shape is [3,1]
                        return self._T.dot(that)
        def __str__(self):
                return f'R={self.R()} | t={self.t()}'

        def R(self):
                return self._T[:2,:2]
        def t(self):
                return self._T[:2,2]
        def angle(self):
                return np.arctan2(self._T[1,0],self._T[0,0])#log(R) R \in SO3
        
        @staticmethod
        def fromPosAngle(x,y,theta):
                return SE2.fromRt(np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]]),np.array([x,y]))
        @staticmethod
        def fromRt(R,t):
                T = np.identity(3)
                T[:2,:2] = R
                T[:2,2] = t
                return SE2(T)
        
        @staticmethod
        def exp(v):
                x,y,theta = v
                print(f'theta={theta}')
                T = np.identity(3)
                T[:2,:2] = np.array([
                        [np.cos(theta), -np.sin(theta)],
                        [np.sin(theta), np.cos(theta)]
                ])  
                if theta < 0.001:
                            #TODO how to handle numerical instability?

                        V = np.zeros((2,2))
                        for i in range(10):
                                theta_2i = (-1)**i*theta**(2*i)*np.identity(2)
                                theta_2i1 = (-1)**i*theta**(2*(i+1))*np.array([[0,-1],[1,0]])
                                V += (theta_2i/(factorial(2*i+1)) + theta_2i1/(factorial(2*i+2)))
                        T[:2,2] = V.dot(np.array([x,y]))
                        #T[:2,2] = (np.array([x,y]))
                else: 
                        V = 1/theta * np.array([
                                [np.sin(theta), -1*(1-np.cos(theta))],
                                [1-np.cos(theta), np.sin(theta)]
                        ])

                        T[:2,2] = V.dot(np.array([x,y]))
                return SE2(T)             
        
        def log(self):
                theta = np.arctan2(self._T[1,0],self._T[0,0])#log(R) R \in SO3
                v = np.zeros((3,))
                #TODO how to handle numerical instability?
                f = theta/(2*(1-np.cos(theta)))
                if not np.isnan(f):
                        v[:2] = f*np.array([[np.sin(theta), 1-np.cos(theta)],
                                           [np.cos(theta)-1,np.sin(theta)]]) @ self.t()
                else:
                        v[:2]=self.t()
                        #V = np.sin(theta)/theta * np.identity(2) + (1 - np.cos(theta))/theta * np.array([[0,1],[-1,0]])
                #v[:2] = np.linalg.inv(V) @ self.t()
                v[2] = theta
                return v
        
        def adjoint(self):
                adj = np.identity(3)
                adj[:2,:2] = self.R()
                adj[0,2] = self.t()[1]
                adj[1,2] = -1*self.t()[0]
                return adj

        def inverse(self):
                return SE2(np.linalg.inv(self._T))

class EKFConstantVelocity:
        def __init__(self, covariance_process, covariance_measurement):
                self._x = np.zeros((6,))
                self._pos = SE2(np.identity(3))
                self._vel = np.zeros((3,))
                self._P = np.identity(6) # Initial uncertainty
                self._Q = covariance_process
                self._R = covariance_measurement
        
        def pose(self):
                return self._pos

        def twist(self):
                return self._vel

        def update(self, y, dt):
                
                # Prediction
                
                motion = SE2.exp(self._vel*dt)
                self._pos = self._pos * motion #self._pos.plus(self._vel.log() * dt,J_x)
                J_f_x = np.zeros((SE2.DoF*2, SE2.DoF*2))
                J_f_x[3:,3:] = np.linalg.inv(motion.adjoint())
                self._P = J_f_x @ self._P @ J_f_x.transpose() + self._Q.transpose()                 

                # expectation
                h = self._vel*dt
                J_h_x = np.hstack([np.zeros((3,3)),np.identity((3))*dt])
                E = J_h_x @ self._P @ J_h_x.transpose()
                print(f"E={E}")


                # innovation
                z = y-h
                Z = E + self._R
                print(f"Z={Z}")

                # print(f"P={P_}")

                # Kalman gain
                K = self._P @ J_h_x.transpose() @ np.linalg.inv(Z)
                print(f"K={K}")

                # Correction
                dx = K @ z
                print(f"dx={dx}")

                self._vel = self._vel + dx[3:]
                #self._pos = self._pos + SE2Tangent(dx[:3])                        
                print(f"vel={self._vel}")

                self._P = self._P - K @ Z @ K.transpose()
                print(f"P={self._P}")
        

                
if __name__ == '__main__':
        state = np.zeros((3,))
        v = np.array([1, 2])
        dt = 0.1
        sigma_measurement = 0.01
        cov_measurement = (sigma_measurement**2) * np.identity(3)
        cov_process = 0.01*np.identity(6)
        kalman = EKFConstantVelocity(cov_process,cov_measurement)
        
        vehicle = DifferentialDrive(0.5)
        n_steps = 100
        trajectory = np.zeros((n_steps, 2))
        trajectory_pred = np.zeros((n_steps, 2))
        velocity = np.zeros((n_steps,3))
        velocity_noisy = np.zeros((n_steps,3))
        velocity_pred = np.zeros((n_steps, 3))

        uncertainty = np.zeros((n_steps, 1))
        pose_gt_prev = None
        for ti in range(n_steps):
                state_prev = state.copy()
                state = vehicle.forward(v,state,dt)
                pose_gt = SE2.fromPosAngle(state[0],state[1],state[2])
                dPose_ = SE2.exp(state_prev).inverse() * SE2.exp(state)
                if ti > 0:
                        dPose_ = (pose_gt_prev.inverse()*pose_gt)
                pose_gt_prev = SE2(pose_gt._T)
                trajectory[ti] = state[:2]
             
                # print(f'x* = {x.transpose()}, x = {state_k}')
                dTwist = dPose_.log()
                dPoseNoisy = dTwist.copy()
                dPoseNoisy[0:2] += sigma_measurement * np.random.rand(2)
                # print(f'dPose = {dPose.transpose()}')
                kalman.update( dPoseNoisy, dt)

                trajectory_pred[ti] = kalman.pose().t()
                velocity_pred[ti] = kalman.twist()
                velocity_noisy[ti] = dPoseNoisy/dt
                velocity[ti] = dPose_.log()/dt
                uncertainty[ti] = np.linalg.det(kalman._P)

        plt.plot(trajectory[:,0],trajectory[:,1])
        plt.plot(trajectory_pred[:,0],trajectory_pred[:,1],'-.')

        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        plt.legend(['GT','Estimated*','Integrated'])

        plt.figure()
        plt.ylabel('v [..]')
        plt.plot(np.linalg.norm(velocity[:,:2],axis=1))
        plt.plot(np.linalg.norm(velocity_noisy[:,:2],axis=1))
        plt.plot(np.linalg.norm(velocity_pred[:,:2],axis=1))
        plt.legend(['GT','Noisy','Estimated'])

        plt.figure()
        plt.ylabel('va [$^\circ$]')
        plt.plot(velocity[:,2])
        plt.plot(velocity_pred[:,2])
        plt.legend(['GT','Estimated'])


        plt.figure()
        plt.ylabel('$\Sigma$')
        plt.plot(uncertainty)

        plt.show()

