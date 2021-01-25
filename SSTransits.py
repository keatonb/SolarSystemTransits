#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 15:39:33 2021

Calculate and plot details of inner Solar System transits as seen from outer
Solar System objects.

Requires package solarsystem (https://pypi.org/project/solarsystem/)
Animation requires imagemagick (imagemagick.org)

@author: keatonb
"""
import solarsystem
import numpy as np
import matplotlib.pyplot as plt
import warnings
from astropy import units as u
from astropy import constants as const
from astropy.time import Time
from datetime import timedelta
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
        self.intransit = ((self.theta < (self.angdiam_Sun + self.angdiam_I)/2.) & 
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
        #The line on this circle makes it look larger than reality,
        #but it's almost too small to see without
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

class Transit:
    """
    Properties and plots of transits in time window.
    
    Calculates:
     - Impact parameter (b)
     TODO: Time of ingress egress
     
    Plots:
     - animate (gif)
     TODO: traceplot (path)
     TODO: lightcurve (simulated)
    
    """
    def __init__(self, innerplanet, outerplanet, starttime, endtime, timestep):
        """
        Parameters:
            innerplanet (str): name of inner planet
            outerplanet (str): name of outer planet
            starttime (datetime): timestamp (UTC) before transit
            endtime (datetime): timestamp (UTC) before transit
            timestep (float): sampling interval (minutes; > 0)
            
        Notes:
            Impact parameter, b, is minimum within timestamp
        """
        #Check that timestep is positive
        if timestep <= 0:
            raise Exception("Timestep must be positive.")
        if timestep > 10:
            warnings.warn("Timesteps longer than 10 minutes may produce poor results")
        deltatime = timedelta(minutes=timestep)
        self.innerplanet = innerplanet
        self.outerplanet = outerplanet
        
        #Compute timestamps
        self.times = [starttime]
        while self.times[-1] < endtime:
            self.times.append(self.times[-1] + deltatime)
        self.mjdtimes = [Time(time).mjd for time in self.times]
        
        #Calculate geometry at each timestamp
        self.geometry = [Geometry(self.innerplanet, self.outerplanet, time)
                         for time in self.times]
        
        #Compute impact parameter (good to timestep precision)
        self.b = np.min([g.theta / ((g.angdiam_Sun)/2.) for g in self.geometry])
        
    def animate(self, filename="Transit.gif", duration=3, figsize=(4,4), dpi=150, **kwargs):
        """Animate the transit
        
        Parameters:
            filename (str): file to save animation to
            duration (float): loop duration (seconds)
            figsize (float,float): width, height in inches
            dpi (float): dots per inch
            **kwargs: for Geometry plot function
        """
        fig,ax = plt.subplots(figsize=figsize)
        
        #No initialization needed
        def init():
            self.geometry[0].plot(ax=ax, show=False, **kwargs)
            plt.tight_layout()
            return 

        #Animation function to call
        def animateframe(i):
            ax.clear() #Clear previous data
            self.geometry[i].plot(ax=ax, show=False, **kwargs)
            return
        
        #Time between frames
        interval = duration/len(self.times)

        #Animate it and save!
        anim = animation.FuncAnimation(fig, animateframe, init_func=init, 
                                       frames=len(self.times), interval=interval, 
                                       blit=False)
        anim.save(filename, dpi=dpi, fps = 1/interval, writer='imagemagick')
    def traceplot(self):
        """Plot path of transit across Sun
        
        COMING SOON
        """
        pass