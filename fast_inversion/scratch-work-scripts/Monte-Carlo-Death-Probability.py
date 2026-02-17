import numpy as np
import math
import matplotlib.pyplot as plt

# Using a probability generating function given by ChatGPT, find the probability
# of a single neutron not reaching the mth generation given a value of p and nu
# 1/p is the probability of fissioning into nu neutrons, both assumed to be constant
# for all neutrons

# return the probability of the branching process having 0 neutrons at generation m
def prob_of_death(m,p,nu):
    if m == 0:
        return 0
    elif m > 0:
        return (p-1)/p + (1/p) * math.pow(prob_of_death(m-1,p,nu), nu)
    
# return the list of probabilities of the branching process having 0 neutrons at generation 0..m
def prob_of_death_list(m,p,nu):
    if m < 0:
        return []
    prob_list = [0]
    for mp in range(1,m+1):
        prob_list.append((p-1)/p + (1/p) * math.pow(prob_list[-1], nu))
    return prob_list


def main():
    p = 3
    nu = 2
    m = 20
    m_list = list(range(m+1))
    death_probs = prob_of_death_list(m,p,nu)
    live_probs = 1-np.array(death_probs)
    plt.plot(range(m+1), death_probs)
    plt.xlabel("Generation m")
    plt.ylabel("Probability of death")
    plt.figure()

    plt.semilogy(m_list, live_probs)
    plt.xlabel("Generation m")
    plt.ylabel("Probability of living")
    #plt.show()

    # find the exponent on the living probability
    log_live_probs = np.log(live_probs)
    plt.plot(m_list, log_live_probs)
    plt.xlabel("Generation m")
    plt.ylabel("Probability of living")
    #plt.show()

    log_slope = log_live_probs[-1] - log_live_probs[-2]
    geom_frac = np.exp(log_slope)
    print("Population scales as ", geom_frac , "^m for generation m")

if __name__ == "__main__":
    main()