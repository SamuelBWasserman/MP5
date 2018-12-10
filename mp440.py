import inspect
import sys

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
    # Your code starts here 
    # You should remove _raise_not_defined() after you complete your code
    # Your code ends here 
    _raise_not_defined()
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


