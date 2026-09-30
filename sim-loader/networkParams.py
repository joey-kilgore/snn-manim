
def getCA3FullScale():
    """Get the parameters needed for creating a config file for the
    full scale CA3 model based on Kopsick et al. 
    """
    groupInfo = {}

    """
    groupInfo is 2 layer dictionary of all neuron parameters and should have all of the
        following keys within each group

        nType - name of neuron
        ei - whether the neuron is excitatory (e) or inhibitory (i)
        pop - the population size of that group
        stimSize - number of neurons in group that receive input current
        stimTime - the duration of that input current
        i - the input current amplitude
        k,a,b,C,d,Vr,Vt,Vmin,Vpeak - 9 param IZ model
    """

    groupInfo[0] = {'nType':'Pyramidal',
                    'ei':'e',
                    'pop':74366,
                    'stimSize':0,
                    'stimTime':0,
                    'i':0.0,
                    'k':0.792,
                    'a':0.008,
                    'b':-42.552,
                    'd':588.0,
                    'C':366.0,
                    'Vr':-63.204,
                    'Vt':-33.604,
                    'Vmin':-38.868,
                    'Vpeak':35.861}
    groupInfo[1] = {'nType':'Axo-axonic',
                'ei':'i',
                'pop':1909,
                'stimSize':0,
                'stimTime':0,
                'i':0.0,
                'k':3.961,
                'a':0.005,
                'b':8.684,
                'd':15.0,
                'C':165.0,
                'Vr':-57.100,
                'Vt':-51.719,
                'Vmin':-73.969,
                'Vpeak':27.799}
    groupInfo[2] = {'nType':'Basket',
                'ei':'i',
                'pop':515,
                'stimSize':0,
                'stimTime':0,
                'i':0.0,
                'k':0.995,
                'a':0.004,
                'b':9.264,
                'd':-6.0,
                'C':45.0,
                'Vr':-57.506,
                'Vt':-23.379,
                'Vmin':-47.556,
                'Vpeak':18.455}
    groupInfo[3] = {'nType':'Basket_CCK+',
                'ei':'i',
                'pop':665,
                'stimSize':0,
                'stimTime':0,
                'i':0.0,
                'k':0.583,
                'a':0.006,
                'b':-1.245,
                'd':54.0,
                'C':135.0,
                'Vr':-58.997,
                'Vt':-39.398,
                'Vmin':-42.771,
                'Vpeak':18.275}
    groupInfo[4] = {'nType':'Bistratified',
                'ei':'i',
                'pop':4631,
                'stimSize':0,
                'stimTime':0,
                'i':0.0,
                'k':3.935,
                'a':0.002,
                'b':16.580,
                'd':19.0,
                'C':107.0,
                'Vr':-64.673,
                'Vt':-58.744,
                'Vmin':-59.703,
                'Vpeak':-9.929}
    groupInfo[5] = {'nType':'Ivy',
                'ei':'i',
                'pop':2334,
                'stimSize':0,
                'stimTime':0,
                'i':0.0,
                'k':1.916,
                'a':0.008,
                'b':12.933,
                'd':45.0,
                'C':364.0,
                'Vr':-70.435,
                'Vt':-40.859,
                'Vmin':-53.400,
                'Vpeak':-6.920}
    groupInfo[6] = {'nType':'MFA_ORDEN',
                'ei':'i',
                'pop':1526,
                'stimSize':0,
                'stimTime':0,
                'i':0.0,
                'k':1.380,
                'a':0.008,
                'b':12.933,
                'd':0.0,
                'C':209.0,
                'Vr':-57.076,
                'Vt':-39.102,
                'Vmin':-40.681,
                'Vpeak':16.313}
    groupInfo[7] = {'nType':'QuadD-LM',
                'ei':'i',
                'pop':3280,
                'stimSize':0,
                'stimTime':0,
                'i':0.0,
                'k':1.776,
                'a':0.006,
                'b':-3.449,
                'd':52.0,
                'C':186.0,
                'Vr':-73.482,
                'Vt':-54.937,
                'Vmin':-64.404,
                'Vpeak':7.066}

    connInfo = {} # (preGroup,postGroup)->(g,probConn)
    connInfo[(0,0)]=(0.3,0.025)
    connInfo[(0,1)]=(0.65,0.015)
    connInfo[(0,2)]=(1.7,0.02)
    connInfo[(0,3)]=(0.85,0.017)
    connInfo[(0,4)]=(0.62,0.016)
    connInfo[(0,5)]=(1.77,0.025)
    connInfo[(0,6)]=(1.1,0.021)
    connInfo[(0,7)]=(1.09,0.013)

    connInfo[(1,0)]=(2.71,0.15)
    
    connInfo[(2,0)]=(2.28,0.15)
    connInfo[(2,1)]=(2.63,0.025)
    connInfo[(2,2)]=(1.8,0.005)
    connInfo[(2,3)]=(1.69,0.005)
    connInfo[(2,4)]=(2.30,0.025)
    connInfo[(2,6)]=(1.36,0.005)
    connInfo[(2,7)]=(1.31,0.005)

    connInfo[(3,0)]=(1.89,0.15)
    connInfo[(3,1)]=(1.94,0.025)
    connInfo[(3,2)]=(0.96,0.005)
    connInfo[(3,3)]=(0.97,0.005)
    connInfo[(3,4)]=(1.78,0.025)
    connInfo[(3,6)]=(1.02,0.005)
    connInfo[(3,7)]=(1.00,0.025)

    connInfo[(4,0)]=(2.08,0.028)
    connInfo[(4,1)]=(2.15,0.007)
    connInfo[(4,2)]=(1.1,0.009)
    connInfo[(4,3)]=(1.44,0.004)
    connInfo[(4,4)]=(2.01,0.033)
    connInfo[(4,5)]=(1.34,0.004)
    connInfo[(4,6)]=(1.57,0.009)
    connInfo[(4,7)]=(1.12,0.008)

    connInfo[(5,0)]=(2.23,0.072)
    connInfo[(5,1)]=(2.29,0.004)
    connInfo[(5,2)]=(1.16,0.016)
    connInfo[(5,3)]=(1.54,0.011)
    connInfo[(5,4)]=(2.16,0.017)
    connInfo[(5,5)]=(1.34,0.004)
    connInfo[(5,6)]=(1.27,0.017)
    connInfo[(5,7)]=(1.18,0.002)

    connInfo[(6,0)]=(1.97,0.042)
    connInfo[(6,1)]=(2.12,0.004)
    connInfo[(6,2)]=(1.08,0.007)
    connInfo[(6,3)]=(1.42,0.005)
    connInfo[(6,4)]=(2.00,0.005)
    connInfo[(6,5)]=(1.35,0.003)
    connInfo[(6,6)]=(1.16,0.002)
    connInfo[(6,7)]=(1.10,0.004)

    connInfo[(7,0)]=(1.72,0.119)
    connInfo[(7,1)]=(1.91,0.005)
    connInfo[(7,2)]=(1.00,0.067)
    connInfo[(7,3)]=(1.31,0.05)

    firingRates = {}
    firingRates[0] = 2.31   # pyramidal
    firingRates[1] = 8.03   # axo-axonic
    firingRates[2] = 13.53  # basket
    firingRates[3] = 4.62   # basket cck+
    firingRates[4] = 6.9    # bistratified
    firingRates[5] = 2.36   # ivy
    firingRates[6] = 2.79   # MFA ORDEN
    firingRates[7] = 7.24   # QuadD-LM

    return groupInfo, connInfo, firingRates