import numpy as np

'''i am guessing what these numbers are, the point is they 
should be labelled clearly with their names'''

SolarEfficiency=0.21
PanelArea=4.6
EnergyConst=SolarEfficiency*PanelArea

def best_velocity(velocities,telemetry_data):
    data= np.array(telemetry_data)
    energies= data[:,1]*data[:,2]*EnergyConst
    #applying the concept of boolean mask
    mask=np.isin(data[:,0],velocities)
    #data[:, 0] is a 1D array, mask also will be same length
    #True where velocity is also in the velocities list, False otherwise
    if not np.any(mask):
        print("No matching velocities found in telemetry data.")
    else:
        bestindex=np.argmax(np.where(mask,energies,-1))
        #np.where: replaces energies with -1 where mask is False, keeps them otherwise
        #np.argmax: gets INDEX of the max value in the np.where thing
        return data[bestindex,0]