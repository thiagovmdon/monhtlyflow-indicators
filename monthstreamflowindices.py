# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 15:28:19 2023

@author: Thiago Nascimento
This code provides the automatic calculation of the Monthly Hydrological Indicators for river streamflow. 
It follows the methodology proposed by Pumo et al. (2018), which is an addaptation of the methodlogy 
proposed for daily streamflow by Richter et al. (1996).

In total there are 22 individual indicators, 5 group indices (MI-HRA), and one Global indice (GMI-HRA).

References: 
    
Pumo, D., Francipane, A., Cannarozzo, M., Antinoro, C., Noto, L.V., 2018. 
Monthly hydrological indicators to assess possible alterations on rivers' flow regime. 
Water Resour. Manag. 32, 3687–3706. https://doi.org/10.1007/s11269-018-2013-6.

Richter, B.D., Baumgartner, J.V., Powell, J., Braun, D.P., 1996. 
A method for assessing hydrologic alteration within ecosystems. 
Conserv. Biol. 10, 1163–1174. https://doi.org/10.1046/j.1523-1739.1996.10041163.x.
"""


from pcraster import *
import numpy as np
from osgeo import gdal, gdalconst
from osgeo import gdal_array
from osgeo import osr
import matplotlib.pylab as plt
import subprocess
import glob,os
import time
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import pymannkendall as mk
import tqdm
from tqdm import tqdm



#%% Firstly one function is defined for the computation of the generic k-th indicator 
# of hydrological alteration:
def pik(Xn25ik, Xn75ik, Xpik):
    if (Xpik >= Xn25ik) and (Xpik <= Xn75ik):
        result = 0
    else:
        if Xn75ik == Xn25ik:
            result = 0
        else:
            result = min(abs((Xpik - Xn25ik)/(Xn75ik - Xn25ik)), abs((Xpik - Xn75ik)/(Xn75ik - Xn25ik)))   
    return np.float16(result)


#%% 
#### Magnitude timing (Group 1):

def p1s_magnitudetiming(datanatural: pd.pandas.core.frame.DataFrame, 
                        datamodified: pd.pandas.core.frame.DataFrame):
    """
    Inputs
    ------------------
    datanatural: dataset[n x y]: 
        dataframe with non-modified (natural) monthly streamflow with the index set as
        datetime. Each column represent a different gauge/grid.
        
    datamodified: dataset[n x y]: 
        dataframe with modified monthly streamflow with the index set as
        datetime. Each column represent a different gauge/grid.
        
    Returns
    --------------------
    p1s: pandas.DataFrame [12 x y] where each row represents a different individual indicator.
    
    MIhra1: pandas.DataFrame [1 x y] where the unique row represents the global group index.
    """
    
    # The number of stations (grids) is computed:
    numstationsused = datanatural.shape[1]-1
    
    # A table to be filled with the p1s is created and filled with NaNs:
    p1s = pd.DataFrame(index = range(1,13), data=np.zeros((12, numstationsused)))
    p1s.iloc[:,:] = np.nan
        
    # A loop is made to compute each indicator. Some metrics are computed inside 
    # the loop to prevent a memory usage error and to save time.
    for j in tqdm(range(numstationsused)):
        for k in range(12):    
            MedianMonthlyStreamflow = datamodified.iloc[:,[j,-1]].groupby('month').median()
            Quantile25MonthlyStreamflow = datanatural.iloc[:,[j,-1]].groupby('month').quantile(q=0.25)
            Quantile75MonthlyStreamflow = datanatural.iloc[:,[j,-1]].groupby('month').quantile(q=0.75)
        
            p1s.iloc[k,j] = pik(Quantile25MonthlyStreamflow.iloc[k,0], Quantile75MonthlyStreamflow.iloc[k,0], MedianMonthlyStreamflow.iloc[k,0])
        
    # Now the first group indice (MI-HRA) is computed:
    MIhra1 = p1s.mean()
    
    return p1s, MIhra1
        
        
        
#%%
#### Magnitude duration (Group 2):
def p2s_magnitudeduration(datanatural: pd.pandas.core.frame.DataFrame, 
                        datamodified: pd.pandas.core.frame.DataFrame):     
    """
    Inputs
    ------------------
    datanatural: dataset[n x y]: 
        dataframe with non-modified (natural) monthly streamflow with the index set as
        datetime. Each column represent a different gauge/grid.
        
    datamodified: dataset[n x y]: 
        dataframe with modified monthly streamflow with the index set as
        datetime. Each column represent a different gauge/grid.
        
    Returns
    --------------------
    p2s: pandas.DataFrame [4 x y] where each row represents a different individual indicator.
    
    MIhra2: pandas.DataFrame [1 x y] where the unique row represents the global group index.
    """

    # The number of stations (grids) is computed:
    numstationsused = datanatural.shape[1]-1    
    
    # This is an empty table to be filled with the indicators for group 2:
    p2s = pd.DataFrame(index = range(4), columns = range(0,numstationsused), data=np.nan)

    # A loop is made to compute each indicator. Some metrics are computed inside 
    # the loop to prevent a memory usage error and to save time.
    for j in tqdm(range(numstationsused)):
    
        Months3flownat = pd.DataFrame(data = datanatural.iloc[:,j].resample('3M',closed='left').sum())
        Months3flowmod = pd.DataFrame(data = datamodified.iloc[:,j].resample('3M',closed='left').sum())
    
        Months6flownat = pd.DataFrame(data = datanatural.iloc[:,j].resample('6M',closed='left').sum())
        Months6flowmod = pd.DataFrame(data = datamodified.iloc[:,j].resample('6M',closed='left').sum())
    
    
        # Statistics for 3-months:
        # Computation of the water years for natural conditions:
        Months3flownat["datetime"] = Months3flownat.index
        Months3flownat['water_year'] = Months3flownat.datetime.dt.year.where(Months3flownat.datetime.dt.month < 10, Months3flownat.datetime.dt.year + 1)
        # Correction of a small bug in the resample:
        Months3flownat['water_year'] = Months3flownat.water_year.where(Months3flownat.datetime.dt.month != 10, Months3flownat.water_year - 1)
        Months3flownat.drop(columns=['datetime'], inplace = True)


        # Computation of the water years for modified conditions:
        Months3flowmod["datetime"] = Months3flowmod.index
        Months3flowmod['water_year'] = Months3flowmod.datetime.dt.year.where(Months3flowmod.datetime.dt.month < 10, Months3flowmod.datetime.dt.year + 1)
        # Correction of a small bug in the resample:
        Months3flowmod['water_year'] = Months3flowmod.water_year.where(Months3flowmod.datetime.dt.month != 10, Months3flowmod.water_year - 1)
        Months3flowmod.drop(columns=['datetime'], inplace = True)

        # Minimum and maximum for 3-months:
        AnnualMin3MonthsFlownat = Months3flownat.groupby('water_year',dropna=False).min()
        AnnualMax3MonthsFlownat = Months3flownat.groupby('water_year',dropna=False).max()

        AnnualMin3MonthsFlowmod = Months3flowmod.groupby('water_year',dropna=False).min()
        AnnualMax3MonthsFlowmod = Months3flowmod.groupby('water_year',dropna=False).max()
    
    
        # Statistics for 6-months:
        # Computation of the water years for natural conditions:
        Months6flownat["datetime"] = Months6flownat.index
        Months6flownat['water_year'] = Months6flownat.datetime.dt.year.where(Months6flownat.datetime.dt.month < 10, Months6flownat.datetime.dt.year + 1)
        # Correction of a small bug in the resample:
        Months6flownat['water_year'] = Months6flownat.water_year.where(Months6flownat.datetime.dt.month != 10, Months6flownat.water_year - 1)
        Months6flownat.drop(columns=['datetime'], inplace = True)

        # Computation of the water years for modified conditions:
        Months6flowmod["datetime"] = Months6flowmod.index
        Months6flowmod['water_year'] = Months6flowmod.datetime.dt.year.where(Months6flowmod.datetime.dt.month < 10, Months6flowmod.datetime.dt.year + 1)
        # Correction of a small bug in the resample:
        Months6flowmod['water_year'] = Months6flowmod.water_year.where(Months6flowmod.datetime.dt.month != 10, Months6flowmod.water_year - 1)
        Months6flowmod.drop(columns=['datetime'], inplace = True)

        # Minimum and maximum for 6-months:
        AnnualMin6MonthsFlownat = Months6flownat.groupby('water_year',dropna=False).min()
        AnnualMax6MonthsFlownat = Months6flownat.groupby('water_year',dropna=False).max()

        AnnualMin6MonthsFlowmod = Months6flowmod.groupby('water_year',dropna=False).min()
        AnnualMax6MonthsFlowmod = Months6flowmod.groupby('water_year',dropna=False).max()
    
        # Computation of the median and quantiles:
        ## First empty data frames for each case are made:
        MedianAnnualMinMax2 = pd.DataFrame(index = ["Min3months","Max3months","Min6months","Max6months"], columns = AnnualMax6MonthsFlownat.columns)
        Quantile25AnnualMinMax2 = pd.DataFrame(index = ["Min3months","Max3months","Min6months","Max6months"], columns = AnnualMax6MonthsFlownat.columns)
        Quantile75AnnualMinMax2 = pd.DataFrame(index = ["Min3months","Max3months","Min6months","Max6months"], columns = AnnualMax6MonthsFlownat.columns)

        # The median is computed only for the modified streamflow:
        MedianAnnualMinMax2.iloc[0,:] = AnnualMin3MonthsFlowmod.median()
        MedianAnnualMinMax2.iloc[1,:] = AnnualMax3MonthsFlowmod.median()
        MedianAnnualMinMax2.iloc[2,:] = AnnualMin6MonthsFlowmod.median()
        MedianAnnualMinMax2.iloc[3,:] = AnnualMax6MonthsFlowmod.median()

        # The quantiles are computed only for the natural streamflow:
        Quantile25AnnualMinMax2.iloc[0,:] = AnnualMin3MonthsFlownat.quantile(q=0.25)
        Quantile25AnnualMinMax2.iloc[1,:] = AnnualMax3MonthsFlownat.quantile(q=0.25)
        Quantile25AnnualMinMax2.iloc[2,:] = AnnualMin6MonthsFlownat.quantile(q=0.25)
        Quantile25AnnualMinMax2.iloc[3,:] = AnnualMax6MonthsFlownat.quantile(q=0.25)

        Quantile75AnnualMinMax2.iloc[0,:] = AnnualMin3MonthsFlownat.quantile(q=0.75)
        Quantile75AnnualMinMax2.iloc[1,:] = AnnualMax3MonthsFlownat.quantile(q=0.75)
        Quantile75AnnualMinMax2.iloc[2,:] = AnnualMin6MonthsFlownat.quantile(q=0.75)
        Quantile75AnnualMinMax2.iloc[3,:] = AnnualMax6MonthsFlownat.quantile(q=0.75)
    
        for k in range(4):
            p2s.iloc[k,j] = pik(Quantile25AnnualMinMax2.iloc[k,0], Quantile75AnnualMinMax2.iloc[k,0], MedianAnnualMinMax2.iloc[k,0])

    # Now the second group indice (MI-HRA) is computed:
    MIhra2 = p2s.mean()
    
    return p2s, MIhra2
    

#%%
#### Timing (Group 3):
def p3s_timing(datanatural: pd.pandas.core.frame.DataFrame, 
                        datamodified: pd.pandas.core.frame.DataFrame):     
    """
    Inputs
    ------------------
    datanatural: dataset[n x y]: 
        dataframe with non-modified (natural) monthly streamflow with the index set as
        datetime. Each column represent a different gauge/grid.
        
    datamodified: dataset[n x y]: 
        dataframe with modified monthly streamflow with the index set as
        datetime. Each column represent a different gauge/grid.
        
    Returns
    --------------------
    p3s: pandas.DataFrame [2 x y] where each row represents a different individual indicator.
    
    MIhra3: pandas.DataFrame [1 x y] where the unique row represents the global group index.
    """

    # The number of stations (grids) is computed:
    numstationsused = datanatural.shape[1]-1
    
    # This is an empty table to be filled with the indicators for group 3:
    p3s = pd.DataFrame(index = range(2), columns = range(0,numstationsused), data=np.nan)
    
    # Computation of the water years for natural condition:
    datanaturalaux = pd.DataFrame(index = datanatural.index)
    datanaturalaux["datetime"] = datanatural.index
    datanaturalaux['water_year'] = datanaturalaux.datetime.dt.year.where(datanaturalaux.datetime.dt.month < 10, datanaturalaux.datetime.dt.year + 1)

    # Computation of the water years for modified condition:
    datamodifiedaux = pd.DataFrame(index = datamodified.index)
    datamodifiedaux["datetime"] = datamodified.index
    datamodifiedaux['water_year'] = datamodifiedaux.datetime.dt.year.where(datamodifiedaux.datetime.dt.month < 10, datamodifiedaux.datetime.dt.year + 1)
    

    # A loop is made to compute each indicator. Some metrics are computed inside 
    # the loop to prevent a memory usage error and to save time.
    for j in tqdm(range(numstationsused)):

    
        # Cliping the data to be used:
        datanatural3 = pd.DataFrame(data= datanatural.iloc[:,j])
        datamodified3 = pd.DataFrame(data= datamodified.iloc[:,j])
    
        # Assigning the water year of each row:
        datanatural3['water_year'] = datanaturalaux['water_year']
        datamodified3['water_year'] = datamodifiedaux['water_year']
    
        # The ID (location) of each minimum or maximum is computed:
        Mininumslocationnatural = datanatural3.groupby('water_year').idxmin()
        Maximumslocationnatural = datanatural3.groupby('water_year').idxmax()
        Mininumslocationmodified = datamodified3.groupby('water_year').idxmin()
        Maximumslocationmodified= datamodified3.groupby('water_year').idxmax()
    
        # Empty tables to be filled with the actual month of each extreme are built:
        # The month of each specific event (maximum and minimum) is computed:

        # Natural:
        MininumslocationMonthnatural = pd.DataFrame(index = Mininumslocationnatural.index, data = Mininumslocationnatural.iloc[:,0].dt.month)
        MaximumlocationMonthnatural = pd.DataFrame(index = Mininumslocationnatural.index, data = Maximumslocationnatural.iloc[:,0].dt.month )

        # Modified:
        MininumslocationMonthmodified = pd.DataFrame(index = Mininumslocationmodified.index, data = Mininumslocationmodified.iloc[:,0].dt.month)
        MaximumlocationMonthmodified = pd.DataFrame(index = Mininumslocationmodified.index, data = Maximumslocationmodified.iloc[:,0].dt.month)
    
        # Replace the months according to the classification of the paper used: (May = 1 until April = 12)
        MininumslocationMonthnatural.replace([5, 6, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], inplace = True)
        MaximumlocationMonthnatural.replace([5, 6, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], inplace = True)
        MininumslocationMonthmodified.replace([5, 6, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], inplace = True)
        MaximumlocationMonthmodified.replace([5, 6, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], inplace = True)

        # Computation of the median and quantiles:
        ## First empty data frames for each case are made:
        # The median(mode) is computed only for the modified streamflow:
        MedianMonths = pd.DataFrame(index = range(2), columns =[0])
        Quantile25Months = pd.DataFrame(index =  range(2), columns =[0])
        Quantile75Months = pd.DataFrame(index = range(2), columns =[0])
    
        # The median(mode) is computed only for the modified streamflow:
        MedianMonths.iloc[0,:] = MininumslocationMonthmodified.mode(dropna=False).max()
        MedianMonths.iloc[1,:] = MaximumlocationMonthmodified.mode(dropna=False).max()


        # The quantiles are computed only for the natural streamflow:
        Quantile25Months.iloc[0,:] = MininumslocationMonthnatural.quantile(q=0.25)
        Quantile25Months.iloc[1,:] = MaximumlocationMonthnatural.quantile(q=0.25)

        Quantile75Months.iloc[0,:] = MininumslocationMonthnatural.quantile(q=0.75)
        Quantile75Months.iloc[1,:] = MaximumlocationMonthnatural.quantile(q=0.75)
    
    
        for k in range(2):
            p3s.iloc[k,j] = pik(Quantile25Months.iloc[k,0], Quantile75Months.iloc[k,0], MedianMonths.iloc[k,0])  
    
    # Now the second group indice (MI-HRA) is computed:
    MIhra3 = p3s.mean()
    
    return p3s, MIhra3


#%%
#### Magnitude frequency (Group 4):
def p4s_timing(datanatural: pd.pandas.core.frame.DataFrame, 
                        datamodified: pd.pandas.core.frame.DataFrame):     
    """
    Inputs
    ------------------
    datanatural: dataset[n x y]: 
        dataframe with non-modified (natural) monthly streamflow with the index set as
        datetime. Each column represent a different gauge/grid.
        
    datamodified: dataset[n x y]: 
        dataframe with modified monthly streamflow with the index set as
        datetime. Each column represent a different gauge/grid.
        
    Returns
    --------------------
    p4s: pandas.DataFrame [2 x y] where each row represents a different individual indicator.
    
    MIhra4: pandas.DataFrame [1 x y] where the unique row represents the global group index.
    """

    # The number of stations (grids) is computed:
    numstationsused = datanatural.shape[1]-1
    
    # This is an empty table to be filled with the indicators for group 4:
    p4s = pd.DataFrame(index = range(2), columns = range(0,numstationsused), data=np.nan)
    
    # Creating empty tables for data filling:
    condlowpulsesnat = pd.DataFrame(index = datanatural.index, columns = range(1), data=np.nan)
    condhighpulsesnat = pd.DataFrame(index = datanatural.index, columns = range(1), data=np.nan)
    condlowpulsesmod = pd.DataFrame(index = datamodified.index, columns = range(1), data=np.nan)
    condhighpulsesmod = pd.DataFrame(index = datamodified.index, columns = range(1), data=np.nan)
    condlowpulsesnat
    # Computing the number of months per year:
    condlowpulsesnat["datetime"] = condlowpulsesnat.index
    condlowpulsesmod["datetime"] = condlowpulsesmod.index
    
    # Computation of water-years:
    condlowpulsesnat['year'] = condlowpulsesnat.datetime.dt.year.where(condlowpulsesnat.datetime.dt.month < 10, condlowpulsesnat.datetime.dt.year + 1)
    condhighpulsesnat['year'] = condlowpulsesnat['year']
    
    condlowpulsesmod['year'] = condlowpulsesmod.datetime.dt.year.where(condlowpulsesmod.datetime.dt.month < 10, condlowpulsesmod.datetime.dt.year + 1)
    condhighpulsesmod['year'] = condlowpulsesmod['year']
    
    
    condlowpulsesnat.drop(columns=['datetime'], inplace = True) 
    condlowpulsesmod.drop(columns=['datetime'], inplace = True) 

    # Loop for computing for each station:
    
    
    for numstations in tqdm(range(numstationsused)):
    
        
        # The quantiles 10% and 90% are computed:
        Quantile10Streamflow = datanatural.iloc[:,numstations].quantile(q=0.10)
        Quantile90Streamflow = datanatural.iloc[:,numstations].quantile(q=0.90)
        
        condlowpulsesnat.iloc[:,0] = np.where((datanatural.iloc[:,numstations] < Quantile10Streamflow),1,0)
        condhighpulsesnat.iloc[:,0] = np.where((datanatural.iloc[:,numstations] > Quantile90Streamflow),1,0)
        condlowpulsesmod.iloc[:,0] = np.where((datamodified.iloc[:,numstations] < Quantile10Streamflow),1,0)
        condhighpulsesmod.iloc[:,0] = np.where((datamodified.iloc[:,numstations] > Quantile90Streamflow),1,0)
        
        # The total number of low and high pulses are computed for each situation:
        lowpulsesnat = condlowpulsesnat.groupby('year',dropna=False).sum()
        highpulsesnat = condhighpulsesnat.groupby('year',dropna=False).sum()
        lowpulsesmod = condlowpulsesmod.groupby('year',dropna=False).sum()
        highpulsesmod = condhighpulsesmod.groupby('year',dropna=False).sum()
    
        # Computation of the median and quantiles:
        ## First empty data frames for each case are made:
        MedianLowAndHighPulses = pd.DataFrame(index = ["lowpulses","highpulses"], columns = lowpulsesmod.columns)
        Quantile25LowAndHighPulses = pd.DataFrame(index = ["lowpulses","highpulses"], columns = lowpulsesnat.columns)
        Quantile75LowAndHighPulses = pd.DataFrame(index = ["lowpulses","highpulses"], columns = lowpulsesnat.columns)
    
        # The median is computed only for the modified streamflow:
        MedianLowAndHighPulses.iloc[0,:] = lowpulsesmod.median()
        MedianLowAndHighPulses.iloc[1,:] = highpulsesmod.median()
    
    
        # The quantiles are computed only for the natural streamflow:
        Quantile25LowAndHighPulses.iloc[0,:] = lowpulsesnat.quantile(q=0.25)
        Quantile25LowAndHighPulses.iloc[1,:] = highpulsesnat.quantile(q=0.25)
    
        Quantile75LowAndHighPulses.iloc[0,:] = lowpulsesnat.quantile(q=0.75)
        Quantile75LowAndHighPulses.iloc[1,:] = highpulsesnat.quantile(q=0.75)
        for k in range(2):
            p4s.iloc[k,numstations] = pik(Quantile25LowAndHighPulses.iloc[k,0], Quantile75LowAndHighPulses.iloc[k,0], MedianLowAndHighPulses.iloc[k,0])
        
    # Now the second group indice (MI-HRA) is computed:
    MIhra4 = p4s.mean()
    
    return p4s, MIhra4        


#%%
#### Frequency rate of change (Group 5):
def p5s_freqrateofchange(datanatural: pd.pandas.core.frame.DataFrame, 
                        datamodified: pd.pandas.core.frame.DataFrame):     
    """
    Inputs
    ------------------
    datanatural: dataset[n x y]: 
        dataframe with non-modified (natural) monthly streamflow with the index set as
        datetime. Each column represent a different gauge/grid.
        
    datamodified: dataset[n x y]: 
        dataframe with modified monthly streamflow with the index set as
        datetime. Each column represent a different gauge/grid.
        
    Returns
    --------------------
    p5s: pandas.DataFrame [2 x y] where each row represents a different individual indicator.
    
    MIhra5: pandas.DataFrame [1 x y] where the unique row represents the global group index.
    """

    # The number of stations (grids) is computed:
    numstationsused = datanatural.shape[1]-1

    # This is an empty table to be filled with the indicators for group 5:
    p5s = pd.DataFrame(index = range(2), columns = range(0,numstationsused), data=np.nan)
    
    
    
    # Loop for computing for each station:
    
    for numstations in tqdm(range(numstationsused)):
        # Cumulative differences: 
        diffnatural = datanatural.iloc[:,numstations].diff(1)
        diffmodififed = datamodified.iloc[:,numstations].diff(1)
        
        # Compute separatly the positive and the negative differences:
        diffnaturalpositives = diffnatural[diffnatural>=0]
        diffnaturalnegatives = diffnatural[diffnatural<0]
        diffmodifiedpositives = diffmodififed[diffmodififed>=0]
        diffmodifiednegatives = diffmodififed[diffmodififed<0]
        
        # Computation of the median and quantiles:
        ## First empty data frames for each case are made:
        MedianDifferences = pd.DataFrame(index = ["positives","negatives"], columns = [0])
        Quantile25Differences = pd.DataFrame(index = ["positives","negatives"], columns = [0])
        Quantile75Differences = pd.DataFrame(index = ["positives","negatives"], columns = [0])
    
        # The median is computed only for the modified streamflow:
        MedianDifferences.iloc[0,:] = diffmodifiedpositives.median(skipna=True)
        MedianDifferences.iloc[1,:] = diffmodifiednegatives.median(skipna=True)
    
        # The quantiles are computed only for the natural streamflow:
        Quantile25Differences.iloc[0,:] = diffnaturalpositives.quantile(q=0.25)
        Quantile25Differences.iloc[1,:] = diffnaturalnegatives.quantile(q=0.25)
    
        Quantile75Differences.iloc[0,:] = diffnaturalpositives.quantile(q=0.75)
        Quantile75Differences.iloc[1,:] = diffnaturalnegatives.quantile(q=0.75)
        
        for k in range(2):
            p5s.iloc[k,numstations] = pik(Quantile25Differences.iloc[k,0], Quantile75Differences.iloc[k,0], MedianDifferences.iloc[k,0])
    
    # Now the second group indice (MI-HRA) is computed:
    MIhra5 = p5s.mean()
    
    return p5s, MIhra5   