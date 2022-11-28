import numpy as np
import pandas as pd


weig=pd.read_csv("weights.csv",header=None)
returns=pd.read_csv("returns.csv",header=None)
isr=pd.read_csv("isr.csv",header=None)

weig=np.array(weig)
returns=np.array(returns)
isr=np.array(isr)

import time
import mybt
start=time.time()
init=1000
a,b,c,d=mybt.backtest(isr,returns,weig,init)
end=time.time()
print(end-start)
print(d.sum())
print(a[0:2,0:5])