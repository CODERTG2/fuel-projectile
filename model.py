import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

class Model:
    """
    A class to model the trajectory of a projectile (ball) launched by a flywheel system,
    accounting for air resistance (drag) and Magnus effect (lift) due to spin.
    """
    def __init__(self, N:float, isTopSpin:bool, theta:float, y0:float, x0=0, D_WHEEL=0.1016, Cd=0.47, Cl=0.2, maxRPM=5700, startRPM=100, deltaRPM=100, deltaT=0.00001, maxTime=3):
        """
        Initialize the projectile model with physical parameters/constants and simulation settings.

        Args:
            N (float): Efficiency factor of the launcher.
            isTopSpin (bool): True if the ball has top spin, False for back spin.
            theta (float): Launch angle in degrees.
            y0 (float): Initial vertical position (height) in meters.
            x0 (float, optional): Initial horizontal position in meters. Defaults to 0.
            D_WHEEL (float, optional): Diameter of the flywheel in meters. Defaults to 0.1016.
            Cd (float, optional): Drag coefficient. Defaults to 0.47.
            Cl (float, optional): Lift coefficient. Defaults to 0.2.
            maxRPM (int, optional): Maximum RPM to simulate. Defaults to 5700.
            startRPM (int, optional): Starting RPM to simulate. Defaults to 100.
            deltaRPM (int, optional): Step size for RPM iteration. Defaults to 100.
            deltaT (float, optional): Time step for the simulation in seconds. Defaults to 0.00001.
            maxTime (float, optional): Maximum duration for the simulation in seconds. Defaults to 3.
        """
        self.N = N # efficiency
        self.spin = 1 if isTopSpin else -1
        self.theta = math.radians(theta) # angle of launch
        self.y0 = y0 # initial height
        self.x0 = x0 # initial distance
        self.R_FUEL = 0.15/2 # radius of the ball
        self.D_WHEEL = D_WHEEL # diameter of the wheel
        
        self.m = 0.20366297 # mass of the ball
        self.g = 9.81 # acceleration due to gravity
        self.p = 1.225 # density of air

        self.Cd = Cd # drag coefficient
        self.Cl = Cl # lift coefficient
        self.A = math.pi*self.R_FUEL**2 # cross-sectional area

        self.maxRPM = maxRPM # maximum RPM
        self.startRPM = startRPM # starting RPM
        self.deltaRPM = deltaRPM # increment in RPM
        self.deltaT = deltaT # time step
        self.maxTime = maxTime # maximum time
    
    def ax(self, vx:float, vy:float, Cl:float):
        """
        Calculate acceleration in the x-direction due to drag and lift.

        Args:
            vx (float): Velocity in x-direction.
            vy (float): Velocity in y-direction.
            Cl (float): Lift coefficient (adjusted for spin).

        Returns:
            float: Acceleration in x-direction.
        """
        return (-1/2 * self.p * self.A / self.m) * (math.sqrt(vx**2 + vy**2)) * (self.Cd*vx + self.spin*Cl*vy)
    def ay(self, vx:float, vy:float, Cl:float):
        """
        Calculate acceleration in the y-direction due to gravity, drag, and lift.

        Args:
            vx (float): Velocity in x-direction.
            vy (float): Velocity in y-direction.
            Cl (float): Lift coefficient (adjusted for spin).

        Returns:
            float: Acceleration in y-direction.
        """
        return -self.g + ((-1/2 * self.p * self.A / self.m) * (math.sqrt(vx**2 + vy**2)) * (self.Cd*vy - self.spin*Cl*vx))
    def vx(self, ax:float, t:float, oldvx:float):
        """
        Calculate new velocity in x-direction.

        Args:
            ax (float): Acceleration in x-direction.
            t (float): Time step duration.
            oldvx (float): Previous velocity in x-direction.

        Returns:
            float: New velocity in x-direction.
        """
        return oldvx + ax * t
    def vy(self, ay:float, t:float, oldvy:float):
        """
        Calculate new velocity in y-direction.

        Args:
            ay (float): Acceleration in y-direction.
            t (float): Time step duration.
            oldvy (float): Previous velocity in y-direction.

        Returns:
            float: New velocity in y-direction.
        """
        return oldvy + ay * t
    def x(self, vx:float, t:float, oldx:float):
        """
        Calculate new position in x-direction.

        Args:
            vx (float): Velocity in x-direction.
            t (float): Time step duration.
            oldx (float): Previous position in x-direction.

        Returns:
            float: New position in x-direction.
        """
        return oldx + vx * t
    def y(self, vy:float, t:float, oldy:float):
        """
        Calculate new position in y-direction.

        Args:
            vy (float): Velocity in y-direction.
            t (float): Time step duration.
            oldy (float): Previous position in y-direction.

        Returns:
            float: New position in y-direction.
        """
        return oldy + vy * t
    
    def simulate(self):
        """
        Simulate the projectile trajectory for a range of RPMs.

        Returns:
            list: A list of dictionaries, where each dictionary contains:
                - "RPM": The RPM value for the simulation run.
                - "Position": A list of (x, y) coordinates over time.
                - "Velocity": A list of (vx, vy) velocities over time.
        """
        data = []
        for RPM in range(self.startRPM, self.maxRPM+1, self.deltaRPM):
            V0 = ((RPM * 2 * math.pi)/60 * self.D_WHEEL/2)/2 * self.N # exit velocity
            W0 = V0 / self.R_FUEL # exit angular velocity
            Cl = self.Cl * self.R_FUEL * W0 / V0 # lift coefficient
            
            vx0 = V0 * math.cos(self.theta)
            vy0 = V0 * math.sin(self.theta)

            ax0 = self.ax(vx0,vy0,Cl)
            ay0 = self.ay(vx0,vy0,Cl)

            velocity = []
            position = []

            curr_x = self.x0
            curr_y = self.y0
            
            for t in np.arange(0,self.maxTime,self.deltaT):
                newvy = self.vy(ay0, self.deltaT, vy0)
                newvx = self.vx(ax0, self.deltaT, vx0)
                
                newy_pos = self.y(vy0, self.deltaT, curr_y)
                newx_pos = self.x(vx0, self.deltaT, curr_x)
                
                velocity.append((newvx, newvy))
                position.append((newx_pos, newy_pos))
                
                ay0 = self.ay(vx0, vy0, Cl)
                ax0 = self.ax(vx0, vy0, Cl)
                
                vy0 = newvy
                vx0 = newvx
                
                curr_x = newx_pos
                curr_y = newy_pos
    
            data.append({
                "RPM": RPM,
                "Position": position,
                "Velocity": velocity
            })
            
        return data

    def table(self, data: list):
        """
        Filter simulation data to find the distance at a specific target condition.
        The condition checks if the ball is at a certain height (approx 1.872m + radius)
        while moving downwards.

        Args:
            data (list): The output from the simulate() method.

        Returns:
            list: A list of dictionaries containing 'RPM' and the 'distance' (x-coordinate)
                  where the condition was met.
        """
        final_table = []
        for d in data:
            counter = 0
            for position in d["Position"]:
                if (position[1] > 1.872 + self.R_FUEL) & (d["Velocity"][counter][1] < 0):
                    final_table.append({
                        'RPM': d['RPM'],
                        'distance': position[0]
                    })
                    break
                counter += 1
        
        return final_table


    def plotRPMDistance(self, data: list):
        """
        Plot the relationship between RPM and Distance.

        Args:
            data (list): The output from the table() method, containing 'RPM' and 'distance'.
        """
        for d in data:
            plt.plot(d['distance'], d['RPM'])
        plt.show()
    
    def regression(self, data: list):
        """
        Perform a curve fit on the RPM vs Distance data to find a mathematical model.

        Args:
            data (list): The output from the table() method.

        Returns:
            tuple: Optimal values for the parameters (a, b, c, d) of the model function.
        """
        x = []
        y = []
        for d in data:
            x.append(d['distance'])
            y.append(d['RPM'])
        
        def model_func(x, a, b, c, d):
            if d == 0: d = 1e-9
            inner = a * x + b
            return (np.sqrt(np.maximum(inner, 0)) + c) / d

        x_data = np.array(x)
        y_data = np.array(y)

        y_sq = (y_data * 1)**2
        slope, intercept = np.polyfit(x_data, y_sq, 1)

        initial_guess = [max(slope, 0.1), intercept, 0, 1]

        low_b = [0, -np.inf, -np.inf, 0.0001]
        high_b = [np.inf, np.inf, np.inf, np.inf]

        popt, pcov = curve_fit(
            model_func, x_data, y_data, 
            p0=initial_guess, 
            bounds=(low_b, high_b),
            method='trf'
        )
            
        plt.scatter(x_data, y_data, label='Data', color='black')
        x_range = np.linspace(min(x_data), max(x_data), 100)
        plt.plot(x_range, model_func(x_range, *popt), 'r-', label='SciPy Fit (Optimized)')
        plt.legend()
        plt.show()

        a, b, c, d = popt
        return a, b, c, d
    
    def run(self):
        """
        Execute the full simulation pipeline: simulate, filter data, plot, and perform regression.

        Returns:
            tuple: The coefficients (a, b, c, d) from the regression analysis.
        """
        table = self.table(self.simulate())
        self.plotRPMDistance(table)
        a, b, c, d = self.regression(table)
        print(f"a={a:.4f}, b={b:.4f}, c={c:.4f}, d={d:.4f}")
        return a, b, c, d