import numpy as np
import matplotlib.pyplot as plt



Cl=[0]
Fl=[3]
Ch=[0]
Fw=[0]
Fcw=[800]
vaug=[1]
vreg=[0.0003]

Ncl=len(Cl)
Nfc=len(Fl)
Nch=len(Ch)
Nfw=len(Fw)
Nfcw=len(Fcw)
Naug=len(vaug)
Nreg=len(vreg)

errors=np.zeros((Ncl,Nfc,Nch,Nfw,Nfcw,Naug,Nreg,400))

for i in range(Ncl):
    for j in range(Nfc):
        for k in range(Nch):
            for l in range(Nfw):
                for m in range(Naug):
                    for n in range(Nreg):
                        for o in range(Nfcw):
                            path="convlayers"+str(Cl[i])+"/fclayer"+str(Fl[j])+"/"+str(Ch[k])+"_channels/width_of_"+str(Fw[l])+"/fcwidth"+str(Fcw[o])+"/reg"+str(vreg[n])+"/aug"+str(vaug[m])
                            errors[i,j,k,l,0,m,n]=np.load(path+"/validation_errors.npy")
np.save("errors",errors)

                            
