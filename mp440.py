import inspect
import sys
import numpy as np

# List used to store previous controls and states
ctl_obs_list = []

'''
Raise a "not defined" exception as a reminder 
'''
def _raise_not_defined():
    print "Method not implemented: %s" % inspect.stack()[1][3]
    sys.exit(1)

'''
Kalman 2D
'''
def kalman2d(data):
    estimated = []
    
    # define coefficients
    A = np.identity(2)
    B = np.identity(2)
    H = np.identity(2)

    # define noise covarience matrices
    Q = np.array([[0.0010, 0.00002],[0.00002, 0.0010]])
    R = np.array([[0.10, 0.005],[0.005, 0.02]])

    # set initial P matrix and initial position
    lamb = 1.0
    x_prev = np.zeros((2,))
    P_prev = lamb*np.identity(2)
    u_prev = np.zeros((2,))

    # time update
    def time_update(x_prev, u_prev, p_prev):
        """ Makes predictions for the kth state and covariance
        Args:
            x_prev: The k-1th state estimate
            u_prev: The k-1th control
            p_prev: The k-1th covariance matrix 
        Returns:
            (x_pred, x_var): The predicted kth state and covariance matrix
        """
        # prediction
        x_pred = np.matmul(A, x_prev) + np.matmul(B, u_prev)
        x_var = np.matmul(np.matmul(H, p_prev), np.transpose(H)) + Q
        return (x_pred, x_var)

    # measurement update
    def measurement_update(x_pred, x_var, z_k):
        """ Computes the updated state estimate x_hat and the updated covariance matrix P(here p_k)
        Args:
            x_hat: The state prediction
            x_var: The covariance predicition
            z_k: The observation/measurement
        Returns:
            (x_hat, p_k): A tuple that has state estimate at k and updated error covriance
        """
        # Kalman gain
        K_k = (np.matmul(x_var, np.transpose(H)))/(np.matmul(np.matmul(H, x_var), np.transpose(H)) + R)

        # measurement
        x_hat = x_pred + np.matmul(K_k, (z_k - np.matmul(H, x_pred)))
        p_k = np.matmul(np.identity(2) - np.matmul(K_k, H), x_var)
        return (x_hat, p_k)

    # loop through data
    for d in data:
        # do the time update
        (x_pred, x_var) = time_update(x_prev, u_prev, P_prev)

        # store previous control
        u_prev = np.array([d[0], d[1]])

        # do the measurement update and store the state and variance
        (x_prev, P_prev) = measurement_update(x_pred, x_var, np.array([d[2], d[3]]))

        # add the state to the list
        estimated.append(x_prev.tolist())
        
    return estimated

'''
Plotting
'''
def plot(data, output):
    # Your code starts here 
    # You should remove _raise_not_defined() after you complete your code
    # Your code ends here 
    _raise_not_defined()
    return

'''
Kalman 2D 
'''
def kalman2d_shoot(ux, uy, ox, oy, reset=False):
    global ctl_obs_list
    # Reset internal data in the first iteration
    if reset is True:
        ctl_obs_list = []

    # Append the current ux, uy, ox, oy to the global list of previous controls and states
    ctl_obs_list .append([ux, uy, ox, oy])

    # Use kalman filter to estimate x and y for each iteration
    # estimate list is of the form [[ex, ey], [ex1, ey1] ,..., [exn, eyn]]
    estimate_list = kalman2d(ctl_obs_list)

    n = len(estimate_list)
    # Fire once there have been at least 150 iterations
    if n > 150:
        decision = (estimate_list[n][0], estimate_list[n][1], True)
        return decision
    else:
        decision = (estimate_list[n][0], estimate_list[n][1], False)
        return decision


'''
Kalman 2D 
'''
def kalman2d_adv_shoot(ux, uy, ox, oy, reset=False):
    global ctl_obs_list
    # Reset internal data in the first iteration
    if reset is True:
        ctl_obs_list = []

    # Append the current ux, uy, ox, oy to the global list of previous controls and states
    ctl_obs_list .append([ux, uy, ox, oy])

    # Use kalman filter to estimate x and y for each iteration
    # estimate list is of the form [[ex, ey], [ex1, ey1] ,..., [exn, eyn]]
    estimate_list = kalman2d(ctl_obs_list)

    n = len(estimate_list)
    # Fire once there have been at least 150 iterations
    if n > 150:
        decision = (estimate_list[n][0], estimate_list[n][1], True)
        return decision
    else:
        decision = (estimate_list[n][0], estimate_list[n][1], False)
        return decision


