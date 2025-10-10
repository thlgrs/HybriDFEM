import matplotlib.pyplot as plt
import h5py
import numpy as np

plt.figure(None, (6, 6))

file = 'NoTension_Lin_Surf.h5'
with h5py.File(file, 'r') as hf:
    # Import what you need
    print(file)
    U_s = hf['U_conv'][-3] * 1000
    P_s = hf['P_r_conv'][-3]

# file = 'NoTension_Lin_Cont.h5'
# with h5py.File(file, 'r') as hf:

#         U_c = hf['U_conv'][-3]*1000
#         P_c = hf['P_r_conv'][-3]

for i in np.arange(2, 11):

    reduction = i / 10
    file = f'NoTension_Lin_Surf_{reduction}.h5'
    with h5py.File(file, 'r') as hf:

        U = hf['U_conv'][-3] * 1000
        P = hf['P_r_conv'][-3]
    if i == 10:
        plt.plot(U, P, label='Equivalent plastic', linewidth=1, color='green')
    plt.plot(U, P, label=None, linestyle=':', linewidth=.5, color='black')

plt.xlim((0, 5))
plt.ylim((0, 1))

plt.plot(U_s, P_s, label='Contact deletion', marker=None, linewidth=1.5)
# plt.plot(U_c, P_c,label='Contact',marker='*')
plt.xlabel('Control Displacement')
plt.ylabel('Load multiplier')
plt.legend()
plt.grid(True)
