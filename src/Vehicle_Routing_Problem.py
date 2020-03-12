import numpy as np  # Useful for array calculations
import math  # used for square roots and powers (normally already installed)
import matplotlib.pyplot as plt  # Needed for plotting the solution
import random  # For random numbers

CustomerCoordinates = np.array([[140.000000, 30.000000],
                                [0.000000, 60.000000],
                                [10.000000, 0.000000],
                                [-100.000000, 100.000000],
                                [-110.000000, 90.000000],
                                [40.000000, 40.000000],
                                [0.000000, 0.000000],
                                [-90.000000, 0.000000],
                                [-60.000000, 110.000000],
                                [-70.000000, 0.000000]])

DepotCoordinates = np.array([[40.000000, 110.000000],
                             [-140.000000, 140.000000],
                             [-10.000000, -10.000000]])

Demand = np.array([1, 1, 2, 2, 1, 1, 2, 1, 8, 2])


class VRP:

    def __init__(self, CustomerCoord, DepotCoord, Demand, WaitingTime, TruckCapacity, nTrucks):

        assert np.sum(Demand) <= nTrucks * TruckCapacity, "More demand than all trucks capacities"
        assert TruckCapacity >= np.max(Demand), "One customer has too much demand that fills more than one customer"

        self.WaitingTime = WaitingTime
        self.TruckCap = TruckCapacity
        self.nCustomers = len(CustomerCoord)
        self.nDepots = len(DepotCoord)
        self.nTrucks = nTrucks
        self.Demand = Demand
        self.CustomerCoord = CustomerCoord
        self.DepotCoord = DepotCoord
        self.Coordinates = np.concatenate((CustomerCoordinates, DepotCoordinates))
        self.OriginalDistmat = np.sqrt(np.sum((self.Coordinates[None, :] - self.Coordinates[:, None]) ** 2, -1))
        self.trucks = []

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
                dem[k] = dem[k] + Demand[trucks[k][s]]
        return (all(i <= self.TruckCap for i in dem))

    def PlotSolution(self):
        plt.clf()
        plt.ion()
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
                    if clusterinfo[k][0] + Demand[
                        c] <= self.TruckCap:  # If capacity is met than assign it to that cluster
                        clusters[k].append([c])
                        clusterinfo[k][0] = clusterinfo[k][0] + Demand[c]
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
        DepProb = 0.2  # Probability that a depot is changed rather than a customers are reallocated
        NoImpr = 20000  # Maximum amount of iterations with no improvement until he stops
        i = 0

        while ticker < NoImpr and (i < runlength):
            Temp = math.exp(-0.005 * i)  # set new temperature
            BestWaitingtime = self.GetTotalWaitingTime()  # Bestwaiting time is current waiting time
            DepOrCust = random.uniform(0, 1)  # Decide if Depot or customers are changed

            if DepOrCust < DepProb:  # if number is smaller than depprob than change random depot
                randk = random.randint(0, self.nTrucks - 1)
                currentDep = self.trucks[randk][0]
                self.trucks[randk][0] = random.randint(self.nCustomers, self.nCustomers + self.nDepots - 1)
            else:  # else change random customer to random place in random truck
                fromk = random.randint(0, self.nTrucks - 1)
                tok = random.randint(0, self.nTrucks - 1)
                if len(self.trucks[
                           fromk]) > 2:  # only do this if there are at least two customers at the from truck. Otherwise results in empty cluster
                    randomC = random.randint(1, len(self.trucks[fromk]) - 1)
                    randomP = random.randint(1, len(self.trucks[tok]) - 1)
                    self.ChangeCustomerOrder(fromk, tok, randomC, randomP)

            # If new solution is better and feasible, update ticker and the solution
            if self.GetTotalWaitingTime() <= BestWaitingtime and self.SolutionFeasible(self.trucks) == True:
                if self.GetTotalWaitingTime() < BestWaitingtime:
                    ticker = 0
                    BestWaitingtime = self.GetTotalWaitingTime()
                    self.PlotSolution()
                    # print("Current Solution is ",BestWaitingtime)
            # else accept the solution with a certain probability. The probability depends on the temperature and how
            # good (bad) the found solution is. If not accepted we change it back
            else:
                rand = BestWaitingtime / self.GetTotalWaitingTime()
                ticker = ticker + 1  # Increase ticker by 1
                if rand > Temp:
                    if DepOrCust < DepProb:
                        self.trucks[randk][0] = currentDep
                    else:
                        self.ChangeCustomerOrder(tok, fromk, randomP, randomC)

            i = i + 1  # Total number or runs

        return self.trucks


game = VRP(CustomerCoord=CustomerCoordinates, DepotCoord=DepotCoordinates, Demand=Demand, WaitingTime=1,
           TruckCapacity=100, nTrucks=4)

game.InitialSolution()
print(game.GetTotalWaitingTime())
game.PlotSolution()

game.SimulatedAnnealing(SA_runlength=5000)
print(game.GetTotalWaitingTime())
game.PlotSolution()
