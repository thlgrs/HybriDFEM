import matplotlib.pyplot as plt
import h5py
import numpy as np

plt.figure(None, (6, 6))

file = 'Coulomb_Lin_Surf.h5'
with h5py.File(file, 'r') as hf:
    # Import what you need
    print(file)
    U_s = hf['U_conv'][-3] * 1000
    P_s = hf['P_r_conv'][-3]

file = 'Coulomb_Lin_Cont.h5'
with h5py.File(file, 'r') as hf:
    U_c = hf['U_conv'][-3] * 1000
    P_c = hf['P_r_conv'][-3]

# %%
# plt.xlim((-7,12))

# plt.ylim((-.8,.8))


plt.plot(U_c, P_c, label='Contact', marker='*')
plt.plot(U_s, P_s, label='Surface', marker='*', linewidth=0)
plt.xlabel('Control Displacement')
plt.ylabel('Load multiplier')
plt.legend()
plt.grid(True)
