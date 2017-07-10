
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plotImage(simg, vmin=None, vmax=None):
    N = simg.faceCount()
    for n in range(N):
        plt.subplot2grid((2, 3), (n / 3, n % 3))
        plt.imshow(simg[n], cmap=plt.cm.get_cmap('jet'), vmin=vmin, vmax=vmax)
        plt.colorbar()
        plt.title('{0}'.format(n))


def addSphere(ax):
    """
    Add a sphere wireframe to the plot
    """

    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = np.cos(u)*np.sin(v)
    y = np.sin(u)*np.sin(v)
    z = np.cos(v)
    ax.plot_wireframe(x, y, z, color="r", alpha=0.15)


def addSolidSphere(ax, radius=1, stride=4, color='b', edgecolor='k', zorder=1):
    """
    Add a solid sphere to the axes
    """

    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)

    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z,  rstride=stride, cstride=stride,
        color=color, edgecolor=edgecolor, zorder=10)


def plotContour(ax, etas, c='k', width=2, plotMarkers=True, zorder=1):
    """
    Plot the countour of a coordinate grid.
    """

    h = etas.shape[0]
    w = etas.shape[1]
    ax.plot(etas[0, :, 0], etas[0, :, 1], etas[0, :, 2], color=c, linewidth=width, zorder=zorder)
    ax.plot(etas[h-1, :, 0], etas[h-1, :, 1], etas[h-1, :, 2], color=c, linewidth=width, zorder=zorder)
    ax.plot(etas[:, 0, 0], etas[:, 0, 1], etas[:, 0, 2], color=c, linewidth=width, zorder=zorder)
    ax.plot(etas[:, w-1, 0], etas[:, w-1, 1], etas[:, w-1, 2], color=c, linewidth=width, zorder=zorder)

    # plot the origin of pixel coordinates
    if plotMarkers:
        ax.scatter(etas[0, 0, 0], etas[0, 0, 1], etas[0, 0, 2], c='r', s=40, zorder=zorder)
        ax.scatter(etas[-1, 0, 0], etas[-1, 0, 1], etas[-1, 0, 2], c='g', s=40, zorder=zorder)
        ax.scatter(etas[0, -1, 0], etas[0, -1, 1], etas[0, -1, 2], c='b', s=40, zorder=zorder)



def plotSurface(ax, etas, stride=4, color='b',
    edgecolor='k', zorder=1, linewidth=1, alpha=1):

    ax.plot_surface(etas[...,0], etas[...,1], etas[...,2],
        rstride=stride, cstride=stride,
        color=color, edgecolor=edgecolor, zorder=10,
        linewidth=linewidth,
        alpha=alpha)


def createFigure(figsize=(10, 10), elev=5, azim=-90, axisoff=False):
    """
    Creates a figure ready for 3D plotting
    """

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev, azim)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    # disables the axis grids
    if axisoff:
        ax.set_axis_off()

    return fig, ax


def scatter(ax, etas, c='b', sampling=1, s=1, zorder=1):

    ax.scatter(etas[::sampling,::sampling,0],
               etas[::sampling,::sampling,1],
               etas[::sampling,::sampling,2], c=c, s=s, zorder=zorder)


# def showSphereCoordinates(etas, c='b', zorder=1):

#     fig = plt.figure(figsize=(10, 10))
#     fig.set_tight_layout(True)
#     ax = fig.add_subplot(111, projection='3d')
#     ax.set_title('Equidistant')
#     ax.view_init(5, -90)
#     ax.set_xlabel('x')
#     ax.set_ylabel('y')
#     ax.set_zlabel('z')
#     ax.scatter(etas[:, :, 0], etas[:, :, 1], etas[:, :, 2], c=c, s=1, zorder=zorder)
#     addSphere(ax)
#     plt.show()


def plotPixelation(pix, ax):
    
    addSphere(ax)
    for n in range(pix.faceCount()):
        etas = pix.faceCoordinates(n)
        plotContour(ax, etas)
