import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from math import pi
from numpy import genfromtxt
from pathlib import Path
import plotly.graph_objects as go
import io
import base64

"""
    Author: Sean Hsu
    Date created: 9/17/2025
This file contains sample code to process XAS and RIXS files.
There are 2 types of XAS output: 
	1. exact peaks (Solver < 4)
	2. Continouos curve (Solver = 4)
only solver = 4 is allowed for mcp for simplicity
"""

def _get_XAS_iter(filedir):
    with open(filedir) as f:
        xas = np.genfromtxt(f,usecols=np.arange(0,2))
    x = xas[1:,0]
    y = xas[1:,1]
    return x,y


def read_dir_xas(dir_name,edge = "L",pol = "XYZ",extension=".txt",solver=4):
    """
    read XAS output file in the specified directory. outputs absorption (x), intensity (y)
    INPUTS:
        dir_name: Directory filepath
        xdata: for exact peaks, xdata can be np.linspace(-25,25,1000) etc
        b: broadening
        edge: absorption edge
        pol: string of "XYZ" for example
    """
    if (solver < 4): 
        ydata = np.zeros(xdata.size)
    for p in pol:
        filename = dir_name+"/XAS_"+edge+"edge_"+p+extension
        if (Path(filename).exists() and Path(filename).is_file()):
            ydata = 0
            x,y = _get_XAS_iter(filename)
            if (ydata == 0):
                ydata = y
            else: ydata = ydata + y
            xdata = x
        else:
            raise RuntimeError(f"XAS file does not exist: {filename}, with polarization: {str(p)}")
    return xdata,ydata


def _get_RIXS_iter(fname,wipe_loss=False):
    """
    Get rixs that are solved by iterative method (BiCGS)
    """
    with open(fname) as f:
        res_raman = np.genfromtxt(f,usecols=np.arange(0,3))
    x = res_raman[1:,0]
    y = res_raman[1:,1]
    z = res_raman[1:,2]
    if (wipe_loss):
        for i in range(x.shape[0]):
            if (x[i] < y[i]): z[i] = 0
    for i in range(x.shape[0]):
        if x[i] != x[0]: break
        if (i == x.shape[0]-1): i = x.shape[0]
    y_dim = int(x.shape[0]/i)
    x = x.reshape((y_dim,i))
    y = y.reshape((y_dim,i))
    z = z.reshape((y_dim,i))
    return x,y,z

def get_RIXS_iter_all(filedir,edge="L",pvin="XYZ",pvout="XYZ",cross=False):
    """
    INPUTS:
        filedir: Directory filepath
        pvin/pvout: Incoming and outgoing polarization
        cross: Cross polarization, if True then X->X, Y->Y and Z->Z are not considered
    """
    xsum = None
    ysum = None
    zsum = None
    for pin in pvin:
        for pout in pvout:
            if (cross and pin == pout): continue
            fname = filedir + "/RIXS_"+edge+"edge_"+pin+"_"+pout+".txt"
            if (Path(fname).exists() and Path(fname).is_file()):
                x,y,z = _get_RIXS_iter(fname)
                xsum = x
                ysum = y
                if (zsum is None): zsum = z
                else: zsum += z
            else:
                raise RuntimeError(f"RIXS file does not exist: {fname}, \
                    with polarization: {str(pin)},{str(pout)}")
    return xsum,ysum,zsum

def set_mpl_style():
    # My personal style preference for matplotlib
    mpl.rcParams['figure.dpi'] = 100
    plt.rcParams.update({'font.size': 25})
    plt.rcParams['axes.linewidth'] = 2.5
    mpl.rcParams['xtick.direction'] = 'in'
    mpl.rcParams['ytick.direction'] = 'in'
    mpl.rcParams['xtick.major.size'] = 10  # Major tick length
    mpl.rcParams['ytick.major.size'] = 10
    mpl.rcParams['xtick.minor.size'] = 5  # Minor tick length
    mpl.rcParams['ytick.minor.size'] = 5
    mpl.rcParams['xtick.major.width'] = 2.5  # Major tick width
    mpl.rcParams['ytick.major.width'] = 2.5
    mpl.rcParams['xtick.minor.width'] = 2  # Minor tick width
    mpl.rcParams['ytick.minor.width'] = 2

def check_pol(s: str):
    s = s.upper()
    allowed = {"X", "Y", "Z"}
    if not (set(s).issubset(allowed) and len(s) == len(set(s))):
        raise RuntimeError(f"Invalid Polarization: {s}")

if __name__ == '__main__':
    _set_mpl_style()
    _check_pol("AXZ")
    exit()
    x, y, z = get_RIXS_iter_all("/Users/seanhsu/Desktop/School/Research/Program File/ED/mcpruns/b95572a68fe9")
    eloss = True

    fig, ax = plt.subplots(figsize=(8,8))
    eloss = True
    if (eloss):
        ax.pcolormesh(x,y,z,cmap="terrain")
        ax.set_ylim([-2,10])
        ax.set_xlim(np.min(x),np.max(x))
        ax.axhline(0,ls="--",c="yellow",lw=2.5)
        ax.set_xlabel("Incident Energy (eV)")
        ax.set_ylabel("Energy Loss (eV)")
    else:
        ax.pcolormesh(x-y,x,z,cmap="terrain")
        lims = np.linspace(-1000,1000,1000)
        plt.plot(lims,lims,ls="--",c="yellow",lw=2.5)
        ax.set_xlim(np.min(x-y),np.max(x-y))
        ax.set_ylim(np.min(x),np.max(x))
        ax.set_ylabel("Incident Energy (eV)")
        ax.set_xlabel("Emission Energy (eV)")

    ax.set_facecolor(plt.cm.terrain(0))
    ax.set_title("RIXS")

    plt.show()
    buf = io.BytesIO()
    dpi = 50
    width_px = fig.get_figwidth() * dpi
    height_px = fig.get_figheight() * dpi
    print(f"Pixel dimensions: {int(width_px)} × {int(height_px)}")
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img_bytes = buf.read()
    plt.close(fig)

    # # Encode to base64 string
    img_base64 = base64.b64encode(img_bytes).decode()
    print(img_base64)
    # with open("test.png","wb") as f:
    #     f.write(base64.b64decode(img_base64))
