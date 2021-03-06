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
from scipy.interpolate import interp1d
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
        
        #Fraction of distance toward Solar limb (0 at center)
        r = self.theta / (self.angdiam_Sun/2.0)
        self.mu = np.sqrt(1-r**2.)
        
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
        ax.set_aspect(1)
        
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
            
def _limbdarkening(phi, u2=0.88, v2=-0.23):
    """limb darkening law

    parameterization from Section 14.7 of Allen's Astrophysical Quantities 
    (4th ed, Cox, 2000, AIP Press)
    default u2,v2 values are for ~V filter @ 600 nm
    phi is angle between solar radius vector and line of sight (radians)
    normalized so disk integrates to 1
    """
    mu = np.cos(phi)
    return (1 - u2 - v2 + u2*mu + v2*(mu**2))/(1-u2/3 - v2/2)

class Transit:
    """
    Properties and plots of transits in time window.
    
    Calculates:
     - MJD (instantaneous and observed) of ingress,egress,midtranist
     - Impact parameter (b)
     
    Plots:
     - animate (gif)
     - traceplot (path)
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
        self.mjd = np.array([Time(time).mjd for time in self.times])
        
        #Calculate geometry at each timestamp
        self.geometry = [Geometry(self.innerplanet, self.outerplanet, time)
                         for time in self.times]
        
        #Get observed times (corrected for light travel time)
        self.mjdobs = self.mjd + np.array([g.timedelay for g in self.geometry])/(24*3600.)
        
        #compute transit start, end, and mid-eclipse times
        #in transit when transitsep <= 1
        transitsep = [g.theta / ((g.angdiam_Sun+g.angdiam_I)/2.0) for g in self.geometry]
        #separate below and after transit
        deepest = np.argmin([g.theta / ((g.angdiam_Sun+g.angdiam_I)/2.) for g in self.geometry])
        #we'll interpolate precise start and end times
        if deepest != 0:
            self.startingress_mjd = float(interp1d(transitsep[:deepest],self.mjd[:deepest],
                                             bounds_error=False)(1))
            self.startingress_mjdobs = float(interp1d(transitsep[:deepest],self.mjdobs[:deepest],
                                             bounds_error=False)(1))
        else:
            self.startingress_mjd = np.nan
            self.startingress_mjdobs = np.nan
        if deepest != len(self.geometry)-1:
            self.endegress_mjd = float(interp1d(transitsep[deepest:],self.mjd[deepest:],
                                          bounds_error=False)(1))
            self.endegress_mjdobs = float(interp1d(transitsep[deepest:],self.mjdobs[deepest:],
                                          bounds_error=False)(1))
        else:
            self.endegress_mjd = np.nan
            self.endegress_mjdobs = np.nan
        self.midtransit_mjd = (self.startingress_mjd + self.endegress_mjd)/2.
        self.midtransit_mjdobs = (self.startingress_mjdobs + self.endegress_mjdobs)/2.
        self.transitdurationobs = (self.endegress_mjdobs - self.startingress_mjdobs)*24*u.h
        
        #Compute geometry at mid-transit
        self.midtransit_geometry = Geometry(self.innerplanet, self.outerplanet, 
                               Time(self.midtransit_mjd,format='mjd').to_datetime())
        #Simulate mid-transit (default limb darkening)
        phi = np.arcsin(2*self.midtransit_geometry.theta/self.midtransit_geometry.angdiam_Sun)
        self.midtransit_depth = ((self.midtransit_geometry.angdiam_I**2/
                                  self.midtransit_geometry.angdiam_Sun**2)*
                                  _limbdarkening(phi))*1e6 # ppm
        
        #Compute impact parameter (good to timestep precision)
        self.b = self.midtransit_geometry.theta / ((self.midtransit_geometry.angdiam_Sun)/2.)
        
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
        
    def traceplot(self, ax=None, fov=(4,4), unit=u.arcsec, show=True, 
                  filename=None, plotsun=True, fontsize=13, **kwargs):
        """Plot path of transit across Sun
        
        Parameters:
            ax (mpl axis): axis to plot to (default: create new fig,ax)
            fov (tuple): (width,height) in solar radii
            unit (astropy angle unit or "solarradii"): unit for axes
            show (bool): whether to show plot (default: True)
            filename (str): filename to save to (default: None)
            sun (bool): plot Sun circle? (default: True)
            fontsize (float): fontsize
            **kwargs: args for figure if no axis provided
        """
        #collect relevant details
        angdiam_I = np.array([g.angdiam_I for g in self.geometry])
        angdiam_Sun = np.array([g.angdiam_Sun for g in self.geometry])
        b = np.array([g.rI[1] for g in self.geometry])
        l = np.array([g.rI[2] for g in self.geometry])
        rI = np.array([g.rI[0] for g in self.geometry])
        rSun = np.array([g.rSun[0] for g in self.geometry])
        
        #Are we plotting in solar radii? (useful for overlaying traces)
        solarradii = unit == "solarradii"
        if solarradii:
            unit = u.radian
        
        #Angular unit conversion (from radians)
        scale = u.radian.to(unit)
        
        #Get trajectory angle, phi, to plot shadow wide enough
        phi = np.arctan(np.diff(b)/np.diff(l))
        phi = np.concatenate((phi,[phi[-1]])) # match length
        
        #Create fig and ax if no ax provided
        if ax is None:
            fig,ax = plt.subplots(**kwargs)
        #Circles must be round
        ax.set_aspect(1)
        
        #Display sun, using angular size at mid-transit (unless solarradii display units)
        midtransit = np.argmin([g.theta / ((g.angdiam_Sun)/2.) for g in self.geometry])
        angdiam_Sun = angdiam_Sun[midtransit]
        sunangrad = scale*angdiam_Sun/2.
        
        if solarradii: #Handle case for solar radii units
            sunangrad = 1
            scale = 2./angdiam_Sun
        if plotsun: #Only plot sun if requested
            sun = plt.Circle((0, 0), sunangrad, color='y', zorder = 1)
            ax.add_patch(sun)
        
        #Is planet in front of Sun?
        infront = rI[midtransit] < rSun[midtransit]
        
        #Display transit path
        linewidth = scale*angdiam_I / np.cos(phi) #Width of shadow path
        ax.fill_between(scale*l,scale*b+linewidth/2.,scale*b-linewidth/2,lw=0, fc='0.2',zorder=2*infront)
        
        ax.set_xlabel(fr"$l'$ ({unit.short_names[0]})", fontsize=fontsize)
        ax.set_ylabel(fr"$b'$ ({unit.short_names[0]})", fontsize=fontsize)
        if solarradii:
            ax.set_xlabel("Solar radii", fontsize=fontsize)
            ax.set_ylabel("Solar radii", fontsize=fontsize)
        
        #Scale axes
        ax.set_xlim(-fov[0]*sunangrad/2, fov[0]*sunangrad/2)
        ax.set_ylim(-fov[1]*sunangrad/2, fov[1]*sunangrad/2)
        
        #Save plot or show
        if filename is not None:
            plt.tight_layout()
            plt.savefig(filename)
        if show:
            plt.tight_layout()
            plt.show()
            
    def simlightcurve(self,limbdarkeningfunc = _limbdarkening, 
                       limbdarkening_args = {"u2":0.88, "v2":-0.23}):
        """
        Simulate transit light curve with limb darkening
        
        Assumes negligible limb darkening gradient across transiting planet disk
        Returns relative model flux at self.mjd_obs
        """
        theta = np.array([g.theta for g in self.geometry])
        angdiam_Sun = np.array([g.angdiam_Sun for g in self.geometry])
        angdiam_I = np.array([g.angdiam_I for g in self.geometry])
        
        #Angle between radial vector and line of sight
        phi = np.arcsin(2*theta/angdiam_Sun)
        
        #compute relative flux
        lc = 1 - (angdiam_I**2/angdiam_Sun**2)*_limbdarkening(phi,**limbdarkening_args)
        lc[np.isnan(lc)] = 1
        return lc