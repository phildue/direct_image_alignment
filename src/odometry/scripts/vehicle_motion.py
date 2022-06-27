
from math import factorial
import numpy as np
import matplotlib.pyplot as plt
from manifpy import SE3,SE2,SE2Tangent

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

class SE2_:
        def __init__(self,T):
                self._T = T

        def __mul__(self,that):
                if type(that) is SE2_:
                        return SE2_(self._T.dot(that._T))
                if type(that) is np.array:
                        assert that.shape is [3,1]
                        return self._T.dot(that)
        def __str__(self):
                return f'R={self.R()} | t={self.t()}'

        def R(self):
                return self._T[:2,:2]
        def t(self):
                return self._T[:2,2]
        
        @staticmethod
        def fromRt(R,t):
                T = np.identity(3)
                T[:2,:2] = R
                T[:2,2] = t
                return SE2_(T)
        
        @staticmethod
        def exp(v):
                x,y,theta = v
                print(f'theta={theta}')
                if theta < 0.001:
                        V = np.zeros((2,2))
                        for i in range(10):
                                theta_2i = (-1)**i*theta**(2*i)*np.identity(2)
                                theta_2i1 = (-1)**i*theta**(2*(i+1))*np.array([[0,-1],[1,0]])
                                V += (theta_2i/(factorial(2*i+1)) + theta_2i1/(factorial(2*i+2)))
                else: 
                        V = 1/theta * np.array([
                                [np.sin(theta), -1*(1-np.cos(theta))],
                                [1-np.cos(theta), np.sin(theta)]
                        ])
                T = np.identity(3)
                T[:2,:2] = np.array([
                        [np.cos(theta), -np.sin(theta)],
                        [np.sin(theta), np.cos(theta)]
                ])  
                T[:2,2] = V.dot(np.array([x,y]))
                return SE2_(T)             
        
        def log(self):
                theta = np.arctan2(self._T[1,0],self._T[0,0])
                A = np.sin(theta)/theta
                B = (1 - np.cos(theta))/theta
                Vinv = 1/(A*A-B*B) * np.array([[A,B],
                                               [-B,A]])

                v = np.zeros((3,))
                if theta > 0:
                        v[:2] = Vinv.dot(self.t())
                else:
                        v[:2] = self.t()
                v[2] = theta
                return v
        
        def adjoint(self):
                adj = np.identity(3)
                adj[:2,:2] = self.R()
                adj[0,2] = self.t()[1]
                adj[1,2] = -1*self.t()[0]
                return adj

        def inverse(self):
                return SE2_(np.linalg.inv(self._T))

class ConstantVelocitySE2:
        def __init__(self):
                pass

        def propagate(self,x,dt):
                """
                 X (+) x*dt = X * exp(x[:3*dt])
                """
                x_ = x.copy()
                pos = x[:3]
                vel = x[3:]
                x_[:3] = (SE2.exp(pos) * SE2.exp(vel*dt)).log()
                return x_
       
        def propagate_jacobian(self,x,dt):
                """
                Jacobian of  X * exp(x[:3*dt])
                """
                J = np.zeros((6,6))
                vel = x[3:]
                J[3:,3:] = np.linalg.inv(SE2.exp(vel*dt).adjoint())
                return J
        
        def f(self,x,dt):
                return self.propagate(x,dt)
        
        def F(self,x,dt):
                return self.propagate_jacobian(x,dt)

        def measurement_model(self,x,dt):
                return SE2.exp(x[3:]*dt).log()

        def measurement_model_jacobian(self,x,dt):
                return np.hstack([np.zeros((3,3)),np.identity(3)])

        def h(self,x,dt):
                return self.measurement_model(x,dt)
        def H(self,x,dt):
                return self.measurement_model_jacobian(x,dt)


class ExtendedKalmanFilter:
        def __init__(self):
                self._system = ConstantVelocitySE2()
                self._x = np.zeros((6,)) # Initial state x,y,theta,vx,vy,vtheta
                self._P = np.identity(6)*1 # Initial uncertainty
                self._Q = np.identity(6) # Process noise
                self._R = np.identity(3)*0.001 # Measurement noise

        def predict(self,dt):
                
                x_ = self._system.f(self._x,dt)
                F  = self._system.F(self._x,dt)
               
                return x_,F.dot(self._P.dot(F.transpose())) + self._Q

        def update(self, z, dt):
                
                x_, P_ = self.predict(dt)

                H = self._system.H(x_, dt) 
                Z = H.dot(P_.dot(H.transpose())) + self._R
                K = P_.dot(H.transpose().dot(np.linalg.inv(Z)))

                self._x = self._x + K.dot(self._innovation(x_,z,dt))
                
                self._P = np.identity(6) - K.dot(H.dot(P_))
                self._P = P - K.dot(Z.dot(K.transpose()))

        def _innovation(self, x, z, dt):
                return (SE2.exp(self._system.h(x,dt)).inverse() * SE2.exp(z)).log()

class ExtendedKalmanFilterManif:
        def __init__(self):
                self._x = np.zeros((6,))
                self._pos = SE2.Identity()
                self._vel = SE2.Identity()
                self._P = np.identity(6) # Initial uncertainty
                self._Q = np.identity(6)*0.1 # Process noise
                self._Q[:3,:3] = 0.001
                self._R = np.identity(3)*1 # Measurement noise
        
        def pose(self):
                return self._pos

        def twist(self):
                return self._vel.log()

        def update(self, y, dt):
                
                # Prediction
                J_x = np.zeros((SE2.DoF, SE2.DoF))
                J_u = np.zeros((SE2.DoF, SE2.DoF))
                
                self._pos = self._pos.plus(self._vel.log(),J_x,J_u)

                self._P[3:,3:] = self._P[3:,3:] + self._Q[3:,3:]
                self._P[:3,:3] = J_x @ self._P[:3,:3] @ J_x.transpose() + self._Q[:3,:3]


                # expectation
                #e = (SE2Tangent(x_[3:]) - SE2Tangent(self._x[3:]/dt))
                e = self._vel.log()
                # TODO Jacobian
                H = np.hstack([np.zeros((3,3)),np.identity((3))])
                E = H @ self._P @ H.transpose()
                print(f"E={E}")


                # innovation
                z = (SE2Tangent(y)-e).coeffs()
                Z = E + self._R
                print(f"Z={Z}")

                # print(f"P={P_}")

                # Kalman gain
                K = self._P @ H.transpose() @ np.linalg.inv(Z)
                print(f"K={K}")

                # Correction
                dx = K @ z
                print(f"dx={dx}")

                self._vel = self._vel + SE2Tangent(dx[3:])
                #self._pos = self._pos + SE2Tangent(dx[:3])                        

                self._P = self._P - K @ Z @ K.transpose()
                print(f"P={self._P}")
                
if __name__ == '__main__':
        kalman = ExtendedKalmanFilterManif()
        state = np.zeros((3,))
        v = np.array([0.1, 0.2])
        dt = 0.1
        vehicle = DifferentialDrive(0.5)
        n_steps = 300
        trajectory = np.zeros((n_steps, 2))
        trajectory_pred = np.zeros((n_steps, 2))
        trajectory_pred2 = np.zeros((n_steps, 2))
        velocity = np.zeros((n_steps,3))
        velocity_pred = np.zeros((n_steps, 3))

        uncertainty = np.zeros((n_steps, 1))

        for ti in range(n_steps):
                state_prev = state.copy()
                state = vehicle.forward(v,state,dt)

                trajectory[ti] = state[:2]
                if ti > 0:
                   trajectory_pred2[ti] = (kalman.twist().exp() * kalman.pose()).translation()
             
                # print(f'x* = {x.transpose()}, x = {state_k}')
                dPose = (SE2(state_prev[0],state_prev[1],state_prev[2]).inverse() * SE2(state[0],state[1],state[2]) ).log().coeffs()
                dPoseNoisy = dPose + np.random.rand(3)/1000
                # print(f'dPose = {dPose.transpose()}')
                kalman.update( dPoseNoisy, dt)
                print(f'i={ti} \nx* = {kalman._x.transpose()}\nx={state}\ndx={dPose}')

                trajectory_pred[ti] = kalman.pose().translation()
                velocity_pred[ti] = kalman.twist().coeffs()/dt
                velocity[ti] = dPose/dt
                uncertainty[ti] = np.linalg.det(kalman._P)

        plt.plot(trajectory[:,0],trajectory[:,1])
        plt.plot(trajectory_pred[:,0],trajectory_pred[:,1],'-o')
        plt.plot(trajectory_pred2[:,0],trajectory_pred2[:,1],'-x')

        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        plt.legend(['GT','Estimated*','Estimated2'])

        plt.figure()
        plt.ylabel('v [..]')
        plt.plot(np.linalg.norm(velocity[:,:2],axis=1))
        plt.plot(np.linalg.norm(velocity_pred[:,:2],axis=1))
        plt.legend(['GT','Estimated'])

        plt.figure()
        plt.ylabel('va [$^\circ$]')
        plt.plot(velocity[:,2])
        plt.plot(velocity_pred[:,2])
        plt.legend(['GT','Estimated'])


        plt.figure()
        plt.ylabel('$\Sigma$')
        plt.plot(uncertainty)

        plt.show()

