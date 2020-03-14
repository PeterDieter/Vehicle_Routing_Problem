import numpy as np  # Useful for array calculations
from src.Vehicle_Routing_Problem import VRP

CustomerCoordinates = np.array([[140.000000, 30.000000],
                                [0.000000, 60.000000],
                                [10.000000, 0.000000],
                                [-100.000000, 100.000000],
                                [-70.000000, 0.000000]])

DepotCoordinates = np.array([[40.000000, 110.000000],
                             [-140.000000, 140.000000],
                             [-10.000000, -10.000000]])

Demand = np.array([1, 1, 2, 2, 1])


game = VRP(CustomerCoord=CustomerCoordinates, DepotCoord=DepotCoordinates, Demand=Demand, WaitingTime=1,
           TruckCapacity=20, nTrucks=2)

game.LinearProgram(LP_runlength=50)

