import matplotlib.pyplot as plt
import numpy


class Painter:
    def __init__(self):
        self.fig = 0
        self.ax = 0
        self.fig2 = 0
        self.ax2 = 0

    def draw(self, xx, data, name, axes, lim=None):
        self.fig, self.ax = plt.subplots()
        self.ax.grid()
        self.ax.set(xlabel=axes[0], ylabel=axes[1])
        if lim:
            self.ax.set_ylim(0.0, 1.5)
        for i in range(len(data)):
            self.ax.plot(xx, data[i])

        plt.legend(name, loc="upper right")

        plt.show()

    def draw_n(self, xx, data, name, axes, lim=None):
        n = len(data)
        self.fig2, self.ax2 = plt.subplots(n, 1)
        plt.subplots_adjust(hspace=1)
        if lim:
            self.ax2[n - 1].set_ylim(-0.1, 1.5)
        for i in range(n):
            self.ax2[i].grid()
            self.ax2[i].plot(xx, data[i])
            self.ax2[i].set_xlabel(axes[0])
            self.ax2[i].set_ylabel(name[i])
        plt.show()

    def draw_uncommon(self, x, data, xname, yname, leg):
        self.fig, self.ax = plt.subplots()
        self.ax.grid()
        self.ax.set(xlabel=xname, ylabel=yname)
        for i in range(len(data)):
            self.ax.plot(x[i], data[i])
        plt.legend(leg, loc="upper center")
        plt.show()

    def draw_classes(self, x, delta, xname, yname):
        self.fig, self.ax = plt.subplots()
        self.ax.set(xlabel=xname, ylabel=yname)
        self.ax.plot(x[:, 0], x[:, 1])
        xmean = numpy.mean(x[:, 1])
        xx = [x[0, 0], x[-1, 0]]
        for i in range(len(delta)):
            self.ax.plot(xx, [delta[i], delta[i]], "r")
        self.ax.plot(xx, [xmean, xmean], "y")
        # self.ax.plot(xx, [0.0, 0.0], "b")

        plt.show()



