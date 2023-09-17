"""
Code for Scientific Computation Project 2
Please add college id here
CID:
"""
import heapq
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import networkx as nx

#===== Codes for Part 1=====#
def part1q1n(Hlist, Hdict={}, option=0, x=[]):
    """
    Code for part 1, question 1
    Hlist should be a list of 2-element lists.
    The first element of each of these 2-element lists
    should be an integer. The second elements should be distinct and >-10000 prior to using
    option=0.
    Sample input for option=0: Hlist = [[8,0],[2,1],[4,2],[3,3],[6,4]]
    x: a 2-element list whose 1st element is an integer and x[1]>-10000
    """
    if option == 0:
        print("=== Option 0 ===")
        print("Original Hlist=", Hlist)
        heapq.heapify(Hlist)
        print("Final Hlist=", Hlist)
        Hdict = {}
        for l in Hlist:
            Hdict[l[1]] = l
        print("Final Hdict=", Hdict)
        return Hlist, Hdict
    elif option == 1:
        while len(Hlist)>0:
            wpop, npop = heapq.heappop(Hlist)
            if npop != -10000:
                del Hdict[npop]
                return Hlist, Hdict, wpop, npop
    elif option == 2:
        if x[1] in Hdict:
            l = Hdict.pop(x[1])
            l[1] = -10000
            Hdict[x[1]] = x
            heapq.heappush(Hlist, x)
            return Hlist, Hdict
        else:
            heapq.heappush(Hlist, x)
            Hdict[x[1]] = x
            return Hlist, Hdict


def part1q2(G,s,x):
    """Input:
    G: weighted NetworkX graph with n nodes (numbered as 0,1,...,n-1)
    s: an integer corresponding to a node in G
    x: an integer corresponding to a node in G
    """

    dinit = np.inf
    Fdict = {}
    Mdict = {}
    n = len(G)
    Plist = [[] for l in range(n)]

    Mdict[s]=1
    Plist[s] = [s]

    while len(Mdict)>0:
        dmin = dinit
        for n,delta in Mdict.items():
            if delta<dmin:
                dmin=delta
                nmin=n
        if nmin == x:
            return dmin, Plist[nmin]
        Fdict[nmin] = Mdict.pop(nmin)
        for m,en,wn in G.edges(nmin,data='weight'):
            if en in Fdict:
                pass
            elif en in Mdict:
                dcomp = dmin*wn
                if dcomp<Mdict[en]:
                    Mdict[en]=dcomp
                    Plist[en] = Plist[nmin].copy()
                    Plist[en].append(en)
            else:
                dcomp = dmin*wn
                Mdict[en] = dcomp
                Plist[en].extend(Plist[nmin])
                Plist[en].append(en)
    return Fdict


def part1q3(G,s,x):
    """Input:
    G: weighted NetworkX graph with n nodes (numbered as 0,1,...,n-1)
    s: an integer corresponding to a node in G
    x: an integer corresponding to a node in G
    Output: Should produce equivalent output to part1q2 given same input
    """

    #Add code here
    dinit = np.inf
    Fdict = {}
    n = len(G)
    Mlist = [[1,s]]
    Mlist, Mdict = part1q1n(Mlist)
    Pdict = {}
    Pdict[s] = [s]
    while len(Mdict)>0:
        dmin = dinit
        Mlist, Mdict, dmin, nmin = part1q1n(Mlist,Hdict = Mdict, option = 1)
        if nmin == x:
            paths = [x]
            last_node = x
            while last_node !=s:
                paths.append(Pdict[last_node])
                last_node = Pdict[last_node]
            return dmin, paths[::-1]
        Fdict[nmin] = dmin
        for m,en,wn in G.edges(nmin,data='weight'):
            if en in Fdict:
                pass
            else:
                if en in Mdict:
                    dcomp = dmin*wn
                    if dcomp<Mdict[en][0]:
                        Mlist, Mdict = part1q1n(Mlist, Hdict=Mdict, option=2, x=[dcomp,en])
                        Pdict[en] = nmin
                else:
                    dcomp = dmin*wn
                    Mlist, Mdict = part1q1n(Mlist, Hdict=Mdict, option=2, x=[dcomp,en])
                    Pdict[en] = nmin
        
    return Fdict

#===== Code for Part 2=====#
def part2q1(n=50,tf=100,Nt=4000,seed=1):
    """
    Part 2, question 1
    Simulate n-individual opinion model

    Input:

    tf,Nt: Solutions are computed at Nt time steps from t=0 to t=tf (see code below)
    seed: ensures same intial condition is generated with each simulation
    Output:
    tarray: size Nt+1 array
    xarray: n x Nt+1 array containing x for the n individuals at
            each time step including the initial condition.
    """
    tarray = np.linspace(0,tf,Nt+1)
    xarray = np.zeros((Nt+1,n))

    def RHS(t,y):
        """
        Compute RHS of model
        """
        #add code here
        #create a matrix with (i,j)th entry equal to y_i-y_j
        A=-np.tile(y.T,(len(y),1))+np.tile(y.T,(len(y),1)).T
        #get f(y_i-y_j) using efficient numpy matrix operations
        B=-A*np.exp(-A**2)
        #sum across js to get the ith entry of the RHS
        return np.mean(B,axis=1) #modify return statement

    #Initial condition
    np.random.seed(seed)
    x0 = n*(np.random.rand(n)-0.5)

    #Compute solution
    out = solve_ivp(RHS,[0,tf],x0,t_eval=tarray,rtol=1e-8)
    xarray = out.y

    return tarray,xarray


def part2q2(n=50): #add input variables if needed
    """
    Add code used for part 2 question 2.
    Code to save your equilibirium solution is included below
    """
    #Add code here
    from scipy import optimize
    def RHS(y):
        """
        Compute RHS of model
        """
        #add code here
        A=-np.tile(y.T,(len(y),1))+np.tile(y.T,(len(y),1)).T
        B=-A*np.exp(-A**2)
        return np.mean(B,axis=1) #modify return statement
    while True:
        x0=np.random.random(50)
        sol=optimize.root(RHS,x0)
        if np.all(abs(sol.x))<=1000 and np.all(abs(sol.x))>0:
            if len(np.unique(sol.x))>=25:
                assert(np.linalg.norm(RHS(sol.x),ord=1)<=10e-15)
                np.savetxt('xeq.txt',sol.x) #saves xeq in file xeq.txt
                return sol.x

def part2q3(): #add input variables if needed
    """
    Add code used for part 2 question 3.
    Code to load your equilibirium solution is included below
    """
    #load saved equilibrium solution
    xeq = np.loadtxt('xeq.txt') #modify/discard as needed

    #Add code here
    #calculate M matrix of the perturbation linearisation
    A=np.tile(xeq.T,(len(xeq),1))-np.tile(xeq.T,(len(xeq),1)).T
    M = np.exp(-(A**2))
    for i in range(len(M)):
        M[i][i]-=np.sum(M[i])
    M=M/len(M)
    #get the eigenvalues and eigenvectors of the perburbation
    l,v = np.linalg.eig(M)
    
    #redefine the differential equation solving function to take in a random perturbation to the equilibrium 
    #as the initial conditions and solve from there
    def part2q1(n=50,tf=100,Nt=4000,seed=1, x_equilibrium=xeq):
    
        tarray = np.linspace(0,tf,Nt+1)
        xarray = np.zeros((Nt+1,n))

        def RHS(t,y):
            """
            Compute RHS of model
            """
            #add code here
            A=-np.tile(y.T,(len(y),1))+np.tile(y.T,(len(y),1)).T
            B=-A*np.exp(-A**2)
            return np.mean(B,axis=1) #modify return statement

        #Initial condition
        np.random.seed(seed)
        x0 = xeq+(np.random.randn(n))/1000000

        #Compute solution
        out = solve_ivp(RHS,[0,tf],x0,t_eval=tarray,rtol=1e-8)
        xarray = out.y

        return tarray,xarray

    tarray,xarray = part2q1(n=50,tf=10000,Nt=4000,seed=1)
    #display each of the xi as they evolve over time after perturbation of the equilibrium
    plt.figure(figsize=(40,30))
    for i in range(len(xarray)):
        plt.subplot(10,5,i+1)
        plt.plot(tarray[:200],[xeq[i] for j in range(200)])
        plt.plot(tarray[:200], xarray[i][:200])
    #return the eigenvalues of M
    return l #modify as needed


def part2q4(n=50,m=100,tf=40,Nt=10000,mu=0.2,seed=1):
    """
    Simulate stochastic opinion model using E-M method
    Input:
    n: number of individuals
    m: number of simulations
    tf,Nt: Solutions are computed at Nt time steps from t=0 to t=tf
    mu: model parameter
    seed: ensures same intial condition is generated with each simulation

    Output:
    tarray: size Nt+1 array
    Xave: size n x Nt+1 array containing average over m simulations
    Xstdev: size n x Nt+1 array containing standard deviation across m simulations
    """

    #Set initial condition
    np.random.seed(seed)
    x0 = n*(np.random.rand(1,n)-0.5)
    X = np.zeros((m,n,Nt+1)) #may require substantial memory if Nt, m, and n are all very large
    X[:,:,0] = np.ones((m,1)).dot(x0)


    Dt = tf/Nt
    tarray = np.linspace(0,tf,Nt+1)
    dW= np.sqrt(Dt)*np.random.normal(size=(m,n,Nt))

    def RHS(t,y):
        """
        Compute RHS of model
        """
        #add code here
        A=-np.tile(y.T,(len(y),1))+np.tile(y.T,(len(y),1)).T
        B=-A*np.exp(-A**2)
        return np.sum(B,axis=1) #modify return statement
    #Iterate over Nt time steps
    for j in range(Nt):
        #Add code here
        X[:,:,j+1] = mu*dW[:,:,j]+Dt*np.array([RHS(0,X[i,:,j]) for i in range(m)])+X[:,:,j]


    #compute statistics
    Xave = X.mean(axis=0)
    Xstdev = X.std(axis=0)

    return tarray,Xave,Xstdev


def part2Analyze(): #add input variables as needed
    """
    Code for part 2, question 4(b)
    """
    #Add code here to generate figures included in your report
    plt.figure(figsize=(40,40))
    plt.subplot(2,2,1)
    tarray,Xave,Xstdev=part2q4(n=50,m=100,tf=50,Nt=10000,mu=0,seed=129)
    for i in range(50):
        plt.plot(tarray,Xave[i,:])
        plt.xlabel(r'$t$')
        plt.ylabel(r'$\overline{X(t)}$')
        plt.grid()
    plt.subplot(2,2,2)
    tarray,Xave,Xstdev=part2q4(n=50,m=100,tf=50,Nt=10000,mu=0.2,seed=129)
    for i in range(50):
        plt.plot(tarray,Xave[i,:])
        plt.xlabel(r'$t$')
        plt.ylabel(r'$\overline{X(t)}$')
        plt.grid()
    plt.subplot(2,2,3)
    tarray,Xave,Xstdev=part2q4(n=50,m=100,tf=50,Nt=10000,mu=0.5,seed=129)
    for i in range(50):
        plt.plot(tarray,Xave[i,:])
        plt.xlabel(r'$t$')
        plt.ylabel(r'$\overline{X(t)}$')
        plt.grid()
    tarray,Xave,Xstdev=part2q4(n=50,m=100,tf=50,Nt=10000,mu=5,seed=129)
    plt.subplot(2,2,4)
    for i in range(50):
        plt.plot(tarray,Xave[i,:])
        plt.xlabel(r'$t$')
        plt.ylabel(r'$\overline{X(t)}$')
        plt.grid()
    plt.show()

    #variance plots
    plt.subplot(1,2,1)
    tarray,Xave,Xstdev=part2q4(n=50,m=100,tf=50,Nt=10000,mu=0.2,seed=129)
    for i in range(50):
        plt.plot(tarray,Xstdev[i,:])
        plt.xlabel(r'$t$')
        plt.ylabel(r'$\overline{X(t)^2}$')
        plt.grid()
    plt.plot(tarray,np.sqrt(tarray)*0.2,linestyle="--",linewidth = "6",label = "theoretical stdev")
    plt.legend()
    plt.subplot(1,2,2)
    tarray,Xave,Xstdev=part2q4(n=50,m=100,tf=50,Nt=10000,mu=5,seed=129)
    for i in range(50):
        plt.plot(tarray,Xstdev[i,:])
        plt.xlabel(r'$t$')
        plt.ylabel(r'$\overline{X(t)^2}$')
        plt.grid()
    plt.plot(tarray,np.sqrt(tarray)*5,linestyle="--",linewidth = "6",label = "theoretical stdev")
    plt.legend()
    plt.show()
    return None #modify as needed
