import numpy as np  # Useful for array calculations
import math  # used for square roots and powers (normally already installed)
import matplotlib.pyplot as plt  # Needed for plotting the solution
import random  # For random numbers
import pulp as plp


class VRP:

    def __init__(self, CustomerCoord, DepotCoord, Demand, WaitingTime, TruckCapacity, nTrucks):

        assert np.sum(Demand) <= nTrucks * TruckCapacity, "More demand than all trucks capacities"
        assert TruckCapacity >= np.max(Demand), "One customer has too much demand that fills more than one customer"
        assert len(Demand) == len(CustomerCoord), "Demand vector and the number of customers is not the same"

        self.WaitingTime = WaitingTime
        self.TruckCap = TruckCapacity
        self.nCustomers = len(CustomerCoord)
        self.nDepots = len(DepotCoord)
        self.nTrucks = nTrucks
        self.Demand = Demand
        self.CustomerCoord = CustomerCoord
        self.DepotCoord = DepotCoord
        self.Coordinates = np.concatenate((CustomerCoord, DepotCoord))
        self.OriginalDistmat = np.sqrt(np.sum((self.Coordinates[None, :] - self.Coordinates[:, None]) ** 2, -1))
        self.trucks = []

        if nTrucks > len(Demand):
            self.nTrucks = len(Demand)

    def GetTotalWaitingTime(self):
        totalwaiting = np.empty((self.nTrucks, 0)).tolist()
        totalwaitingtime = np.zeros(self.nTrucks)
        for k in range(0, self.nTrucks):
            for s in range(0, len(self.trucks[k]) - 1):
                totalwaiting[k].append([self.OriginalDistmat[self.trucks[k][s], self.trucks[k][s + 1]]])

        for k in range(0, self.nTrucks):
            totalwaitingtime[k] = np.sum(np.cumsum(totalwaiting[k]))

        return self.WaitingTime * np.sum(totalwaitingtime)

    # Changes the customer order in a truck.
    def ChangeCustomerOrder(self, fromtruck, totruck, frm, to):
        selected_Customer = self.trucks[fromtruck][frm]
        del self.trucks[fromtruck][frm]
        self.trucks[totruck].insert(to, selected_Customer)

        return self.trucks

    # Check if the solution is Feasible (capacity constraint is met for each truck)
    def SolutionFeasible(self, trucks):
        dem = np.zeros(self.nTrucks)
        for k in range(0, self.nTrucks):
            for s in range(1, len(trucks[k]) - 1):
                dem[k] = dem[k] + self.Demand[trucks[k][s]]
        return (all(i <= self.TruckCap for i in dem))

    def PlotSASolution(self):

        for k in range(0, self.nTrucks):
            for s in range(0, len(self.trucks[k]) - 1):
                x = [self.Coordinates[self.trucks[k][s], 0], self.Coordinates[self.trucks[k][s + 1], 0]]
                y = [self.Coordinates[self.trucks[k][s], 1], self.Coordinates[self.trucks[k][s + 1], 1]]
                plt.plot(x, y, color="black")

        # Make customer points blue
        for i in range(0, self.nCustomers):
            plt.scatter(self.Coordinates[i, 0], self.Coordinates[i, 1], color="blue")
            # plt.annotate('C = %i' %(i),(Coord[i,0]+1,Coord[i,1]+3))

        # Make Depot points red
        for i in range(self.nCustomers, self.nCustomers + self.nDepots):
            plt.scatter(self.Coordinates[i, 0], self.Coordinates[i, 1], color="red")
            # plt.annotate('Dep = %i' %(i+1-N),(Coord[i,0]+1,Coord[i,1]+3))
        
        plt.show()

    def InitialSolution(self):
        ### Begin With Computing Clusters ###
        Distmat = np.copy(self.OriginalDistmat)
        sortedDem = np.argsort(self.Demand)[::-1]  # Sort the Demand
        RandomCentroids = []  # Initialize a vector for centroids (Even though it says random its not random)
        for i in range(0, self.nTrucks):  # For every truck
            x = self.CustomerCoord[sortedDem[i], 0]  # Get the x-coordinate of the first customer in sorted list
            y = self.CustomerCoord[sortedDem[i], 1]  # Get the y-coordinate of the first customer in sorted list
            RandomCentroids.append([x, y])  # These coordinates are now the first centroids

        RandomCentroids = np.array(RandomCentroids)  # Convert it to np.array
        RandomCentroids2 = RandomCentroids + 1  # So that we enter the while loop

        # While centroids are not the same, enter the while loop
        while np.array_equal(RandomCentroids2, RandomCentroids) == False:
            RandomCentroids2 = RandomCentroids  # Set Centroids2 to the Centroids
            clusters = np.empty((self.nTrucks, 0)).tolist()  # Initialize Clusters as list
            clusterinfo = []  # Intialize an array of clusters information (Coordinates, Demand etc.)
            for k in range(0, self.nTrucks):
                clusterinfo.append(
                    [0, 0, RandomCentroids[k, 0], RandomCentroids[k, 1]])  # Update the coordinates to clusterinfo

            # Calculate the Distance from each customer to the Centriods. First Initialize this distance matrix
            DistCentroids = np.zeros((self.nTrucks, self.nCustomers))
            for c in range(0, self.nTrucks):
                for k in range(0, self.nCustomers):
                    Dist = math.sqrt(
                        math.pow(self.Coordinates[k, 0] - RandomCentroids[c, 0], 2) + math.pow(
                            self.Coordinates[k, 1] - RandomCentroids[c, 1], 2))
                    DistCentroids[c, k] = Dist

            # Assign the customers to the clusters. Keeping in mind the capacity constraint. Start with the ones with highest Demand
            for c in sortedDem:
                assigned = False  # All are not assigned in the beginning
                while assigned == False:
                    MinCentroid = np.min(DistCentroids[:, c])  # Check the nearest centroid to that customer
                    index = np.where(DistCentroids[:, c] == MinCentroid)  # Get the Index of that centroid
                    k = index[0][0]  # k  is the truck number
                    if clusterinfo[k][0] + self.Demand[
                        c] <= self.TruckCap:  # If capacity is met than assign it to that cluster
                        clusters[k].append([c])
                        clusterinfo[k][0] = clusterinfo[k][0] + self.Demand[c]
                        clusterinfo[k][1] = clusterinfo[k][1] + MinCentroid
                        clusterinfo[k][2] = clusterinfo[k][2] + self.CustomerCoord[c, 0]
                        clusterinfo[k][3] = clusterinfo[k][3] + self.CustomerCoord[c, 1]
                        assigned = True
                    else:
                        DistCentroids[k, c] = 100000  # else set the distance very high

            # Update the cluster information and the coordinates of the Centroids (for every cluster) (Average x and y Coordinates)
            for k in range(0, self.nTrucks):
                if len(clusters[k]) > 0:
                    clusterinfo[k][2] = (clusterinfo[k][2] - RandomCentroids[k, 0]) / len(clusters[k])
                    clusterinfo[k][3] = (clusterinfo[k][3] - RandomCentroids[k, 1]) / len(clusters[k])
                    RandomCentroids[k, 0] = clusterinfo[k][2]
                    RandomCentroids[k, 1] = clusterinfo[k][3]

        ### Assign customer to trucks. One truck per cluster ###
        # We do this with the nearest neighbor algorithm as described in chapter 1 but only within each cluster.

        for k in range(0, self.nTrucks):

            # First we find the closest depot to any of the customers
            bestdist = 1000
            bestd = 0
            for i in range(0, len(clusters[k])):
                for d in range(0, self.nDepots):
                    currdist = Distmat[d + self.nCustomers, np.sum(clusters[k][i])]
                    if currdist < bestdist:
                        bestdist = currdist
                        bestd = d + self.nCustomers
                        bestc = np.sum(clusters[k][i])
            self.trucks.append([bestd, bestc])
            Distmat[:, bestc] = 10000  # Mark this customer as visited

            # Now we apply the nearest neighbor heuristic for the rest
            for j in range(0, len(clusters[k]) - 1):
                currentcust = np.sum(
                    self.trucks[k][len(self.trucks[k]) - 1])  # current customer is the last customer in the list
                bestdist = 10000
                for c in range(0, len(clusters[k])):
                    currdist = Distmat[
                        currentcust, clusters[k][
                            c]]  # Calculate the distance between the current customer and another customer
                    if currdist < bestdist:  # Find the closest customer
                        bestc = np.sum(clusters[k][c])
                        bestdist = currdist
                Distmat[:, bestc] = 10000  # mark the best customer as visited
                self.trucks[k].append(bestc)  # add him to list

        return self.trucks

    def SimulatedAnnealing(self, SA_runlength):
        ### Begin Simulated Annealing ###
        ticker = 0  # Iteration after no improvement was found
        runlength = SA_runlength  # Maximum number of runs
        DepProb = 0.05  # Probability that a depot is changed rather than a customers are reallocated
        NoImpr = 20000  # Maximum amount of iterations with no improvement until he stops
        i = 0

        while ticker < NoImpr and (i < runlength):
            Temp = math.exp(-0.001 * i)  # set new temperature
            BestWaitingtime = self.GetTotalWaitingTime()  # Bestwaiting time is current waiting time
            DepOrCust = random.uniform(0, 1)  # Decide if Depot or customers are changed

            if DepOrCust < DepProb:  # if number is smaller than depprob than change random depot
                if self.nTrucks > 1:
                    randk = np.random.randint(0, self.nTrucks - 1)
                    currentDep = self.trucks[randk][0]
                    self.trucks[randk][0] = random.randint(self.nCustomers, self.nCustomers + self.nDepots - 1)
            else:  # else change random customer to random place in random truck
                fromk = random.randint(0, self.nTrucks - 1)
                tok = random.randint(0, self.nTrucks - 1)
                ncust = len(self.trucks[fromk])
                if ncust > 2:  # only do this if there are at least two customers at the from truck. Otherwise results in empty cluster
                    randomC = random.randint(1, len(self.trucks[fromk]) - 1)
                    randomP = random.randint(1, len(self.trucks[tok]) - 1)
                    self.ChangeCustomerOrder(fromk, tok, randomC, randomP)

            # If new solution is better and feasible, update ticker and the solution
            if self.GetTotalWaitingTime() <= BestWaitingtime and self.SolutionFeasible(self.trucks) == True:
                if self.GetTotalWaitingTime() < BestWaitingtime:
                    ticker = 0
                    BestWaitingtime = self.GetTotalWaitingTime()
                    # print("Current Solution is ",BestWaitingtime)
            # else accept the solution with a certain probability. The probability depends on the temperature and how
            # good (bad) the found solution is. If not accepted we change it back
            else:
                rand = np.random.rand()
                ticker = ticker + 1  # Increase ticker by 1
                if rand > Temp:
                    if DepOrCust < DepProb:
                        self.trucks[randk][0] = currentDep
                    else:
                        if ncust > 2:
                            self.ChangeCustomerOrder(tok, fromk, randomP, randomC)

            i = i + 1  # Total number or runs

        return self.trucks

    def LinearProgram(self, LP_runlength):
        Places = np.arange(self.nCustomers + self.nDepots)  # Number of Places (Customers and Depots)
        Trucks = np.arange(self.nTrucks)  # Number of Trucks
        M = 100000 # big M constraint

        # Define the Problem
        prob = plp.LpProblem("Min Waiting", plp.LpMinimize)

        # Create Decision variables
        Routes = [(i, j, k) for i in Places for j in Places for k in Trucks]
        # A dictionary called route_vars is created to contain the referenced variables (the routes)
        route_vars = plp.LpVariable.dicts("Route", (Places, Places, Trucks), 0, None, cat=plp.LpInteger)
        u = plp.LpVariable.dicts("Help variable for MTZ constraint", indexs=((i, k) for i in Places for k in Trucks),
                                 lowBound=0, cat=plp.LpContinuous)
        K_used = plp.LpVariable("Trucks that are used", cat=plp.LpInteger)
        TotalCosts = plp.LpVariable("Total costs", cat=plp.LpContinuous)

        # Optimization Function
        prob += plp.lpSum([u[i, k] for i in Places for k in Trucks]) * self.WaitingTime, "Optimization function"

        # constraint 1: All customers must be visited
        for i in range(0, len(Places) - self.nDepots):
            prob += plp.lpSum([route_vars[j][i][k] for k in Trucks for j in Places if i != j]) == 1

        # constraint 2: All truck that arrive must also leave at any Place
        for h in Places:
            for k in Trucks:
                prob += plp.lpSum([route_vars[i][h][k] for i in Places]) - plp.lpSum(
                    [route_vars[h][j][k] for j in Places]) == 0

        # Constraint 3: Truck Capacity cannot be exceeded
        for k in Trucks:
            prob += plp.lpSum([self.Demand[i] * route_vars[i][j][k] for i in range(0, len(Places) - self.nDepots) for j in Places]) <= self.TruckCap

        # constraint 4: Every truck can leave a depot at most one time
        for k in Trucks:
            prob += plp.lpSum(
                [route_vars[i][j][k] for i in range(len(Places) - self.nDepots, len(Places)) for j in range(0, len(Places))]) <= 1

        # constraint 5: Every truck can arrive at a depot at most one time
        for k in Trucks:
            prob += plp.lpSum(
                [route_vars[i][j][k] for j in range(len(Places) - self.nDepots, len(Places)) for i in range(0, len(Places))]) <= 1

        # constraint 6: Total Number of Trucks
        prob += plp.lpSum(
            [route_vars[i][j][k] for i in range(len(Places) - self.nDepots, len(Places)) for j in range(0, len(Places) - self.nDepots) for k
             in Trucks]) <= K_used

        # constraint 7: No subtours (Miller Tucker Zemlin)
        for i in Places:
            for j in range(0, self.nCustomers):
                for k in Trucks:
                    if j != i:
                        prob += u[i, k] - u[j, k] + self.OriginalDistmat[i, j] <= M * (1 - route_vars[i][j][k])

        # Constraint 8: Total trucks used need to be smaller or equal to total amount of trucks
        prob += K_used <= self.nTrucks

        # Solve the problem. Set the maximum time that the solver will run
        prob.solve(plp.PULP_CBC_CMD(maxSeconds=LP_runlength))

        # Print Status of the solver as well as objective value found so far
        print("status:", plp.LpStatus[prob.status])
        print("Total Waiting Time in hours: ' ", plp.value(prob.objective))

        # Print and Store Results
        Lfrom = []  # List with Places from i
        Lto = []  # List with Places to j
        Ltruck = []  # List of truck used to go from place i to j
        trucks = []
        for i in Places:
            for k in Trucks:
                for j in Places:
                    if plp.value(route_vars[i][j][k]) != 0 and plp.value(route_vars[i][j][k]) is not None:
                        if i < self.nCustomers and j < self.nCustomers:
                            Lfrom.append(i)
                            Lto.append(j)
                            Ltruck.append(k)
                        if i >= self.nCustomers:
                            Lfrom.append(i)
                            Lto.append(j)
                            Ltruck.append(k)


        # Plot the solution (First lines, than points)
        plt.figure()
        for s in range(0, len(Lfrom)):
            x = [self.OriginalDistmat[Lfrom[s], 0], self.OriginalDistmat[Lto[s], 0]]
            y = [self.OriginalDistmat[Lfrom[s], 1], self.OriginalDistmat[Lto[s], 1]]
            truck = Ltruck[s]
            c = [float(truck) / float(10), 0.0, float(10 - truck) / float(10)]
            plt.plot(x, y, color=c)
            plt.gray()

        # Make customer points blue
        for i in range(0, self.nCustomers):
            plt.scatter(self.OriginalDistmat[i, 0], self.OriginalDistmat[i, 1], color="blue")
            # plt.annotate('C = %i' %(i+1),(Coord[i,0]+1,Coord[i,1]+3))

        # Make Depot points red
        for i in range(self.nCustomers, self.nCustomers + self.nDepots):
            plt.scatter(self.OriginalDistmat[i, 0], self.OriginalDistmat[i, 1], color="red")
            # plt.annotate('Dep = %i' %(i+1-N),(Coord[i,0]+1,Coord[i,1]+3))

        plt.show()


