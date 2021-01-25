#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 15:39:33 2021

Calculate and plot details of inner Solar System transits as seen from outer
Solar System objects.

@author: keatonb
"""
import solarsystem
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy import constants as const
from datetime import datetime,timedelta
from matplotlib import animation

#from NASA Planetary Fact Sheet #km
#https://nssdc.gsfc.nasa.gov/planetary/factsheet/
planetdiameter = {"Mercury":4879,
                  "Venus":12104,
                  "Earth":12756,
                  "Mars":6792,
                  "Jupiter":142984,
                  "Saturn":120536,
                  "Uranus":51118,
                  "Neptune":49528}

class Geometry:
    """
    innerplanet relative to Sun as seen from outerplanet at time
    
    Derived in Bell & Rustamkulov (2021, in prep.)
    
    """
    def __init__(self, innerplanet, outerplanet, time):
        """
        Parameters:
            innerplanet (str): name of inner planet
            outerplanet (str): name of outer planet
            time (datetime): timestamp (UTC)
        """
        self.innerplanet = innerplanet.capitalize()
        self.outerplanet = outerplanet.capitalize()
        self.time = time
        
        #Get heliocentric ecliptic planet positions at time: 
        #(Longitude (l, deg), Latitude (b, deg), Distance (r, AU))
        H = solarsystem.Heliocentric(year=time.year, month=time.month, 
                                     day=time.day, hour=time.hour, 
                                     minute=time.minute + time.second/60 + time.microsecond/1e6)
        planets = H.planets()
        
        #Get heliocentric ecliptic planet positions: 
        #(Longitude (l, deg), Latitude (b, deg), Distance (r, AU))
        O = planets[outerplanet]
        I = planets[innerplanet]
        E = planets["Earth"]

        #Convert to spherical position vector: 
        #[r (AU, b (radians), l (radians)]
        rvec = lambda P: np.array([P[2],P[1]*np.pi/180,P[0]*np.pi/180]) 
        rO = rvec(O)
        rI = rvec(I)
        rE = rvec(E)

        #Convert to Cartesian coordinates (x,y,z in AU)
        xvec = lambda rP: np.array([rP[0]*np.cos(rP[1])*np.cos(rP[2]),
                                    rP[0]*np.cos(rP[1])*np.sin(rP[2]),
                                    rP[0]*np.sin(rP[1])])
        xO = xvec(rO)
        xI = xvec(rI)
        xE = xvec(rE)

        #Get positions relative to outer planet
        xO_Sun = - xO
        xO_I = xI - xO
        xO_E = xE - xO

        #Align x-axis with Sun for relative planet positions
        #With two rotation matrices: x' = BAx
        A = np.array([[-np.cos(rO[2]),-np.sin(rO[2]),0],
                      [np.sin(rO[2]),-np.cos(rO[2]),0],
                      [0,0,1]])
        B = np.array([[np.cos(rO[1]),0,-np.sin(rO[1])],
                      [0,1,0],
                      [np.sin(rO[1]),0,np.cos(rO[1])]])
        BA = np.matmul(B, A)

        xvecprime = lambda xO_P: np.matmul(BA,xO_P)
        xO_Sun_prime = xvecprime(xO_Sun) #Passes a sanity check
        xO_I_prime = xvecprime(xO_I)
        xO_E_prime = xvecprime(xO_E)
        self.xSun = xO_Sun_prime
        self.xI = xO_I_prime
        self.xE = xO_I_prime

        #Convert back to spherical coordinates
        #for on-sky positions as seen from O [r (AU,b (radians),l (radians)]
        rvecprime = lambda xvecp: np.array([np.sqrt(np.sum(xvecp**2.)),
                                            np.arctan(xvecp[2]/np.sqrt(np.sum(xvecp[:2]**2))),
                                            -np.arctan(xvecp[1]/xvecp[0])])
        rO_Sun_prime = rvecprime(xO_Sun_prime) #Passes a sanity check
        rO_I_prime = rvecprime(xO_I_prime) #Passes a sanity check
        rO_E_prime = rvecprime(xO_E_prime) #Praise Boas!
        self.rSun = rO_Sun_prime
        self.rI = rO_I_prime
        self.rE = rO_I_prime
        
        #Angular separation between inner planet and Sun (radians)
        self.theta = np.arccos(np.dot(xO_Sun_prime,xO_I_prime)/(rO_Sun_prime[0]*rO_I_prime[0]))

        #Angular diameters of inner planet and Sun (radians)
        self.angdiam_Sun = 2*const.R_sun.to(u.AU)/(rO_Sun_prime[0]*u.AU)
        self.angdiam_I = planetdiameter[innerplanet]*u.km.to(u.AU)/rO_I_prime[0]

        #Are we in transit?
        self.transit = ((self.theta < (self.angdiam_Sun + self.angdiam_I)/2.) & 
                        (rO_I_prime[0] < rO_Sun_prime[0]))

        #Light travel time delay to Earth (seconds)
        self.timedelay = ((rO_I_prime[0] + rO_E_prime[0])*u.AU/const.c).to(u.s).value

    def plot(self, ax=None, fov=(4,4), unit=u.arcsec, show=True, 
             filename=None, timedelay=True, fontsize=13, **kwargs):
        """
        Plot snapshot of Sun, innerplanet from outerplanet
        
        Parameters:
            ax (mpl axis): axis to plot to (default: create new fig,ax)
            fov (tuple): (width,height) in solar radii
            unit (astropy angle unit): unit for axes
            show (bool): whether to show plot (default: True)
            filename (str): filename to save to (default: None)
            timedelay (bool): add light-travel time to text?
            fontsize (float): fontsize
            **kwargs: args for figure if no axis provided
        """
        #Create fig and ax if no ax provided
        if ax is None:
            fig,ax = plt.subplots(**kwargs)
        #Circles must be round
        ax.set_aspect(fov[1]/fov[0])
        
        #Angular unit conversion (from radians)
        scale = u.radian.to(unit)
        
        #Display sun, planet
        sunangrad = scale*self.angdiam_Sun/2.
        sun = plt.Circle((0, 0), sunangrad, color='y', zorder = 1)
        #Is planet in front of Sun?
        infront = self.rI[0] < self.rSun[0]
        planet = plt.Circle((scale*self.rI[2], scale*self.rI[1]), 
                            scale*self.angdiam_I/2., color='blue',
                            zorder=2*infront)
        ax.add_patch(sun)
        ax.add_patch(planet)

        #Add text
        time = self.time
        if timedelay:
            time += timedelta(seconds=self.timedelay)
        ax.text(0.03,0.02,(f"{self.innerplanet} from {self.outerplanet} \n" +
                           time.strftime('%Y-%m-%d %H:%M:%S')),
                transform=ax.transAxes, ha='left', va='bottom', fontsize=fontsize)
        ax.set_xlabel(fr"$l'$ ({unit.short_names[0]})", fontsize=fontsize)
        ax.set_ylabel(fr"$b'$ ({unit.short_names[0]})", fontsize=fontsize)
        #Scale axes
        ax.set_xlim(-fov[0]*sunangrad/2, fov[0]*sunangrad/2)
        ax.set_ylim(-fov[1]*sunangrad/2, fov[1]*sunangrad/2)
        
        #Save plot or show
        if filename is not None:
            plt.savefig(filename)
        if show:
            plt.show()