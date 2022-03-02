#!/usr/bin/env python

import math
import numpy as np
from scipy.integrate import simps


#******************************************************************
#SCRIPT TO COMPUTE NUMBER OF IONISING PHOTONS PRODUCED BY STELLAR EVOLUTION MODELS
#need to read model output to get effective temperature and luminosity values through evolution ZAMS to end He-b, call these temp and lum
#******************************************************************


def planck(wave,T):  #wave in cm and T in K
    #constants
    h_constant=6.6260755e-27 #cgs: erg s
    k_constant=1.380658e-16  #cgs: erg K^-1
    c_constant=2.9979246e10  #cgs: cm s^-1
    a1=2*h_constant*(c_constant**2) #2hc^2
    a2=h_constant*c_constant/k_constant #hc/k
    #planck function
    B1=a1*(wave**(-5))
    T=10**T
    B2=wave*T
    B3=np.float128(a2/B2)
    bbflux=B1*((np.exp(B3)-1)**(-1))
    return bbflux

def radflux(bbflux):
    radflux=bbflux*math.pi  #integrating over solid angle
    return radflux

def radius(T,L): #calculate radius from L and T
    l_sol=3.8427e33 #cgs, ergs/s
    l_val = 10**L
    l_fin=l_val*l_sol
    t_val = 10**T
    Rsol=6.9551e10 #cgs
    sigma = 5.6704e-5
    rsq=(l_val)/(4*(np.pi)*sigma*(t_val**4)) #still need to multiply by l_sol, used l_val rather than l_fin to avoid Inf values
    radius = np.sqrt(rsq*l_sol)
    return radius

#*******************************************************************

class Nphotons:

    def __init__(self, T, L):
        self.T=T
        self.L=L

        #wavelengths in angstrom
        wavelengths=np.arange(6000.0)+1
        #wavelengths in cm
        obs_wave=wavelengths*1e-8

        #finding threshold wavelength index
        edge_hyd=912e-8
        edge_he1=504e-8
        edge_he2=228e-8
        index_edge_hyd=np.where(obs_wave==edge_hyd)
        index_edge_he1=np.where(obs_wave==edge_he1)
        index_edge_he2=np.where(obs_wave==edge_he2)

        #calculating radiative flux
        bbflux=planck(obs_wave,T)
        fobs=radflux(bbflux)

        #calculating # of ionising photons
        rad=radius(T,L)
        constant=6.326062e16  #4*!pi/(h*c) (cgs)
        log_2=np.log10(constant)+np.log10(rad**2) #4*pi/(h*c) r^2

        #for Hydrogen
        func=fobs[0:index_edge_hyd[0][0]]*obs_wave[0:index_edge_hyd[0][0]]
        dx=obs_wave[0:index_edge_hyd[0][0]]
        nphot_hyd=log_2+np.log10(simps(func,dx))
        self.nphot_hyd=nphot_hyd

        #for HeI
        func=fobs[0:index_edge_he1[0][0]]*obs_wave[0:index_edge_he1[0][0]]
        dx=obs_wave[0:index_edge_he1[0][0]]
        nphot_he1=log_2+np.log10(simps(func,dx))
        self.nphot_he1=nphot_he1

        #for HeII
        func=fobs[0:index_edge_he2[0][0]]*obs_wave[0:index_edge_he2[0][0]]
        dx=obs_wave[0:index_edge_he2[0][0]]
        nphot_he2=log_2+np.log10(simps(func,dx))
        self.nphot_he2=nphot_he2


#********************************************************************

#replace these input values to read arrays of temp and lum from stellar evol model output
temp=float(input('insert effective temperature (log(T) (K)): '))
lum=float(input('insert luminosity (log(L/Lsun)): '))

#can loop over values of temp and lum to find Nphotons at each timestep
model1=Nphotons(temp,lum)

print('H photons: ',10**model1.nphot_hyd,' photons/s')
print('HeI photons: ',10**model1.nphot_he1,' photons/s')
print('HeII photons: ',10**model1.nphot_he2,' photons/s')
