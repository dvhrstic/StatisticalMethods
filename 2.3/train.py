
import numpy as np

# This file contains example codes for generating map and simulating train moves.  

# This function creates the experiment environment.
#
# n: grid dimension. Map is initialized as (n x n) grid, then modified. 
# saveOpt: Saves Adjacency Matrix and Vertex Settings into .txt file.
# displayOpt: Displays the values of matrices etc. on the screen.
# 
# Adjacency Matrix (A): has (V x V) dimensionality, where V is the number of vertices in the map. 
# A_{i,j} represents the connection between vertex i and vertex j. 
# If A_{i,j} = 0, vertex i and vertex j are not connected.
# If A_{i,j} = 1, vertex i and vertex j are connected by "L" edge on vertex i's side.
# If A_{i,j} = 2, vertex i and vertex j are connected by "R" edge on vertex i's side.
# If A_{i,j} = 1, vertex i and vertex j are connected by "0" edge on vertex i's side.
# Note that A_{i,j} does not have to be same with A_{j,i}. The labels of edges might be different. 
# Vertex Settings: has (V x 1) dimensionality. It represent the default switch settings for each switch. 1 for "L", 2 for "R".
def createExperiment(n, saveOpt=True, displayOpt=False):
    print("*** Creating the map and switch settings for n=%d ***\n" % n)
    
    # Initialize (n x n) grid
    adjMatrix = adjMatrixInit(n)

    # Modify the adjacency until it is valid (every vertex has degree 3)
    while(checkConnections(adjMatrix)==False):
        adjMatrix = combineDegree2(adjMatrix)  # Combine degree 2 vertices 
        adjMatrix = separateDegree4(adjMatrix) # Separate degree 4 vertices
    
    if displayOpt:
        print("Connectivity Matrix")
        print(adjMatrix)
        print("Marginal of Connectivity Matrix")
        marg = marginalize(adjMatrix)
        print(marg)
    
    # Assign edges to the graph "1-L, 2-R or 3-0"
    adjMatrix = assignEdges(adjMatrix).astype(int)
    # Initialize switch settings "1-L, 2-R or 3-0"
    vertexSettings = assignVertexSettings(adjMatrix).astype(int)
    
    if displayOpt:
        print("Adjacency Matrix Dimensions")
        print(adjMatrix.shape)
        print("Adjacency Matrix")
        print(adjMatrix)
        print("Vertex Settings")
        print(vertexSettings)
        print("Vertex Count")
        print(vertexSettings.shape)
        
    # Save to the file
    if saveOpt:
        filename = "exp_%d_adjacency.txt" % n
        np.savetxt(filename, adjMatrix, fmt='%d')

        filename = "exp_%d_vertexsettings.txt" % n
        np.savetxt(filename, vertexSettings, fmt='%d')
    
    print("*** The map and switch settings are created  ***\n")
    return adjMatrix, vertexSettings

# Helper Function. Checks i and j values are whether within the grid dimensions or not. 
def isValid(i,j,gridDim):
    if i>= 0 and i<gridDim and j>= 0 and j<gridDim:
        result = 1
    else:
        result = 0
    return result

# This function initializes (n x n) grid.
# Note that (n x n) grid has (n+1) vertices.
def adjMatrixInit(n):
    gridDim = n+1
    numVer = gridDim * gridDim

    adjMatrix = np.zeros((numVer,numVer))

    # This section sets A_{i,j} = 1 if vertex i and vertex j are connected 
    # For each vertex, we are looking for its neighbours
    for i in range(gridDim):
        for j in range(gridDim):
            idVer = i * gridDim + j

            # Connect A_{i,j} and A_{i-1,j}
            prop_i = i-1
            prop_j = j
            if isValid(prop_i,prop_j,gridDim) == 1:
                prop_id = prop_i * gridDim + prop_j
                adjMatrix[idVer,prop_id] = 1

            # Connect A_{i,j} and A_{i+1,j}
            prop_i = i+1
            prop_j = j
            if isValid(prop_i,prop_j,gridDim) == 1:
                prop_id = prop_i * gridDim + prop_j
                adjMatrix[idVer,prop_id] = 1

            # Connect A_{i,j} and A_{i,j-1}
            prop_i = i
            prop_j = j-1
            if isValid(prop_i,prop_j,gridDim) == 1:
                prop_id = prop_i * gridDim + prop_j
                adjMatrix[idVer,prop_id] = 1

            # Connect A_{i,j} and A_{i,j+1}
            prop_i = i
            prop_j = j+1
            if isValid(prop_i,prop_j,gridDim) == 1:
                prop_id = prop_i * gridDim + prop_j
                adjMatrix[idVer,prop_id] = 1

    return adjMatrix

# This function combines degree 2 edges of the Adjacency Matrix.
def combineDegree2(adjMatrix):
    marg = marginalize(adjMatrix)
    remVer = np.where(marg == 2)[0]

    # Connects neighbours to each other
    for i in range(len(remVer)):
        connVer = np.where(adjMatrix[remVer[i],:] == 1)[0]
        adjMatrix[connVer[0],connVer[1]] = 1
        adjMatrix[connVer[1],connVer[0]] = 1

    # Removes unnecessary vertices
    for i in range(len(remVer)):
        delVer = remVer[len(remVer)-i-1]
        temp = np.delete(adjMatrix,delVer,0)
        temp2 = np.delete(temp,delVer,1)
        adjMatrix = temp2

    return adjMatrix

# This function separates degree 4 edges of the Adjacency Matrix.
def separateDegree4(adjMatrix):
    marg = marginalize(adjMatrix)

    divVer = np.where(marg == 4)[0]

    for i in range(len(divVer)):
        # Add new column and row for the new vertex
        adjMatrix = np.column_stack((adjMatrix, np.zeros(adjMatrix.shape[0])))
        adjMatrix = np.row_stack((adjMatrix, np.zeros(adjMatrix.shape[1])))

        curVer = divVer[i]
        newVer = adjMatrix.shape[0]-1

        # Connect original vertex with the new vertex
        adjMatrix[curVer,newVer] = 1
        adjMatrix[newVer,curVer] = 1

        # Transfer 2 neighbors to the new vertex from original vertex
        ind = np.where(adjMatrix[curVer]==1)[0][0:2]
        adjMatrix[newVer,ind[0]]=1
        adjMatrix[ind[0],newVer]=1
        adjMatrix[newVer,ind[1]]=1
        adjMatrix[ind[1],newVer]=1

        # Remove transferred neighbours from original vertex
        adjMatrix[curVer,ind[0]]=0
        adjMatrix[ind[0],curVer]=0
        adjMatrix[curVer,ind[1]]=0
        adjMatrix[ind[1],curVer]=0

    return adjMatrix

# This function checks whether each vertex has degree 3. 
def checkConnections(adjMatrix): 
    marg = marginalize(adjMatrix)
    result = np.array_equal(marg, 3*np.ones(marg.shape[0])) # True if equal
    return result

# This function marginalizes Adjacency Matrix.
# Note that instead of summing each row, this function counts non-zero elements at each row. 
# [0 0 3 1 0 2] and [0 0 1 1 0 1] will give the same result; 3. 
def marginalize(adjMatrix):
    marg = np.zeros(adjMatrix.shape[0])
    for i in range(adjMatrix.shape[0]):
        conn = np.where(adjMatrix[i]>0)[0]
        marg[i] = len(conn)
    return marg

# This function assignes edge information to Adjacency Matrix. 
# Initially, every row of Adjacency Matrix has three 1's. This function randomly switches non-zero values with 1,2 and 3.
# 1 represents "L", 2 represents "R" and 3 represents "0" edge. 
def assignEdges(adjMatrix):
    edges = np.array([1,2,3])
    
    for i in range(adjMatrix.shape[0]):
        np.random.shuffle(edges)
        conn = np.where(adjMatrix[i]==1)[0]
        
        adjMatrix[i,conn[0]] = edges[0]
        adjMatrix[i,conn[1]] = edges[1]
        adjMatrix[i,conn[2]] = edges[2]
    
    return adjMatrix

# This function initializes switch settings. 
# 1 represents "L", 2 represents "R".
# Default probability of having "R" edge is 0.5 
def assignVertexSettings(adjMatrix, pRight=0.5):
    settings = np.ones(adjMatrix.shape[0])
    
    for i in range(len(settings)):
        u = np.random.rand()
        if u <= pRight:
            settings[i] = 2
    return settings

# This function calculates next position and exiting direction of the train. 
# curPos, curDir: Train's current position and exiting direction of current position.
def findNextMove(adjMatrix, vertexSettings, curPos, curDir):
    # Finds next position by looking at Adjacency Matrix.
    nextPos = np.where(adjMatrix[curPos]==curDir)[0][0]
    
    # Finds next position's incoming edge by looking at Adjacency Matrix.
    inEdge = adjMatrix[nextPos, curPos]
    
    # If the train comes from "L" or "R" edge to the nextPos, it exits through "0" edge. 
    if inEdge == 1 or inEdge == 2:
        nextDir = 3
    # If the train comes from "0" edge to the nextPos, it checks switch settings to determine exiting edge. 
    else: 
        nextDir = vertexSettings[nextPos]
    
    return nextPos, nextDir
    
# This function simulates a train's movement given the map, switch settings.
# numIter: Total number of moves the train made. Set to 20 as default.
# pErr: Probability of observing the wrong signal from switches. Set to 0.05 as default.
# saveOpt: Saves the real and observed signals into .txt files. Note that the file name contains starting position and direction of the train. 
def simulate(adjMatrix, vertexSettings, numIter=20, pErr=0.05, saveOpt=True, displayOpt=False):
    print("*** Simulating the train for %d iterations with noise probability %.5f with %d switches ***\n" % (numIter, pErr, adjMatrix.shape[0]))
    startStats = []
    
    realSignals = np.zeros(numIter)
    observedSignals = np.zeros(numIter)
    
    for i in range(numIter):
        if i == 0:
            # Randomly select starting position and direction of the train
            curPos = np.random.randint(adjMatrix.shape[0])
            curDir = 1 + np.random.randint(3) # 1, 2 or 3
            
            startStats.append(curPos)
            startStats.append(curDir)
        else:
            curPos = nextPos
            curDir = nextDir
          
        realSignals[i] = curDir
        
        # Add noise to the wrong signal observations
        u = np.random.rand()
        if u <= pErr:
            if u <= pErr/2:
                if curDir == 3:
                    curSignal = 1
                else:
                    curSignal = curDir + 1
            else:
                if curDir == 1:
                    curSignal = 3
                else:
                    curSignal = curDir - 1
        else:
            curSignal = curDir
            
        observedSignals[i] = curSignal
        
        if displayOpt:
            print("Iter %d. Current pos: %d, real direction: %d. Signal: %d" % (i, curPos, curDir, curSignal))
        
        # Finds which switch the train goes next and calculates the exiting direction
        nextPos, nextDir = findNextMove(adjMatrix, vertexSettings, curPos, curDir)
        
    if displayOpt:
        print("Real Signals:")
        print(realSignals)
        
        print("Observed Signals:")
        print(observedSignals)
        
    # Saves real and observed (noisy) signals into .txt file
    if saveOpt:
        filename = "simulate_numIter_%d_pErr_%d_startPos_%d_startDir_%d_observedsignals.txt" % (numIter, pErr, startStats[0], startStats[1])
        np.savetxt(filename, observedSignals, fmt='%d')
        
        filename = "simulate_numIter_%d_pErr_%d_startPos_%d_startDir_%d_realsignals.txt" % (numIter, pErr, startStats[0], startStats[1])
        np.savetxt(filename, realSignals, fmt='%d')
        
    print("*** Train simulation finished ***\n")
    return observedSignals.astype(int), realSignals.astype(int)

def main():
    print("This \"train.py\" file contains example functions to create a valid map and simulating train moves.")
    print("For details, check the source file and Train_Sketch.pdf for toy example.\n")
    print("In order to create the map, use the following function:")
    print("adjMatrix, vertexSettings = createExperiment(n=2, saveOpt=True, displayOpt=False) \n")
    print("After creating the map, simulate the train by:")
    print("observedSignals, realSignals = simulate(adjMatrix, vertexSettings, numIter=5, pErr=0.05, saveOpt=True, displayOpt=False)")
    
    
if __name__ == "__main__": main()

adjMatrix, vertexSettings = createExperiment(n=2, saveOpt=True, displayOpt=False) 
observedSignals, realSignals = simulate(adjMatrix, vertexSettings, numIter=5, pErr=0.05, saveOpt=True, displayOpt=False)


