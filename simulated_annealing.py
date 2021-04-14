# Import some other libraries that we'll need
# matplotlib and numpy packages must also be installed
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import random
import math

# define objective function
random.seed(1234)
min_val = 5
max_val = 14
delta_x = 1
def f(k):
    d = {11:358.28,13:160.28,15:135.83,17:105.27,19:130.74,21:221.63,23:250.26,25:261.72,27:290.78,29:260.17} #ODD VALUES 2K+1, 5,6,7,8,9,10,11,12,13,14
    '''
    Objective takes input LM,EC tool and uncorrected Subsample
    Outputs Perplexity
    return obj   
    '''
    return d[2*k+1]
# Start location
x_start = random.randint(min_val,max_val)

##################################################
# Simulated Annealing
##################################################
# Number of cycles
n = 9
# Number of trials per cycle
m = 3
# Number of accepted solutions
na = 0.0
# Probability of accepting worse solution at the start
p1 = 0.7
# Probability of accepting worse solution at the end
p50 = 0.001 ## END PROB NOT NECESSARILY AT 50, T50 IS JUS A NAME
# Initial temperature
t1 = -1.0/math.log(p1)
# Final temperature
t50 = -1.0/math.log(p50) ## END TEMPERATURE NOT NECESSARILY AT 50, IT'S JUST A NAME
# Fractional reduction every cycle
frac = (t50/t1)**(1.0/(n-1.0))
# Initialize x
x = np.zeros(n+1)

x[0] = x_start

# xi = np.zeros(2)
xi = x_start
na = na + 1.0
# Current best results so far
# xc = np.zeros(2)
xc = x[0]
fc = f(xi)
fs = np.zeros(n+1)
fs[0] = fc
# Current temperature
t = t1
# DeltaE Average
DeltaE_avg = 0.0
for i in range(n):
    print('Cycle: ' + str(i) + ' with Temperature: ' + str(t))
    for j in range(m):
        # Generate new trial points
        xi = random.randint(max(min_val,xc-delta_x),min(max_val,xc+delta_x))
        # xi[1] = xc[1] + random.random() - 0.5
        # Clip to upper and lower bounds
        # xi[0] = max(min(xi[0],1.0),-1.0)
        # xi[1] = max(min(xi[1],1.0),-1.0)
        DeltaE = abs(f(xi)-fc)
        if (f(xi)>fc):
            # Initialize DeltaE_avg if a worse solution was found
            #   on the first iteration
            if (i==0 and j==0): DeltaE_avg = DeltaE
            # objective function is worse
            # generate probability of acceptance
            p = math.exp(-DeltaE/(DeltaE_avg * t))
            # determine whether to accept worse point
            if (random.random()<p):
                # accept the worse solution
                accept = True
            else:
                # don't accept the worse solution
                accept = False
        else:
            # objective function is lower, automatically accept
            accept = True
        if (accept==True):
            # update currently accepted solution
            xc = xi

            fc = f(xc)
            # increment number of accepted solutions
            na = na + 1.0
            # update DeltaE_avg
            DeltaE_avg = (DeltaE_avg * (na-1.0) +  DeltaE) / na
    # Record the best x values at the end of every cycle
    x[i+1] = xc
    fs[i+1] = fc
    # Lower the temperature for next cycle
    t = frac * t

# print solution
print('Best solution: ' + str(2*xc+1))
print('Best objective: ' + str(fc))

# plt.plot(x[:,0],x[:,1],'y-o')
# plt.savefig('contour.png')

# fig = plt.figure()
# ax1 = fig.add_subplot(211)
# ax1.plot(fs,'r.-')
# ax1.legend(['Objective'])
# ax2 = fig.add_subplot(212)
# ax2.plot(x[:,0],'b.-')
# ax2.plot(x[:,1],'g--')
# ax2.legend(['x1','x2'])

# # Save the figure as a PNG
# plt.savefig('iterations.png')

# plt.show()
