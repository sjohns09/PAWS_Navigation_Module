import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import os

class Plot:

    def __init__(self, title):
        style.use('fivethirtyeight')
        self.title = title
        self.x_data = []
        self.y_data = []
    
    def add_point(self, x_data_point, y_data_point):
        self.x_data.append(x_data_point)
        self.y_data.append(y_data_point)        
    
    def plot(self, filename):
        self.fig = plt.figure()
        self.fig.suptitle(self.title)
        plt.plot(self.x_data, self.y_data)
        self.fig.savefig(filename)
        plt.close(self.fig)
        