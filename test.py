import numpy as np
import matplotlib.pyplot as plt

# knum=9
# a=1
# k1=np.linspace(-np.pi/a,np.pi/a,num=knum+1)
# k2=np.roll(k1,1)
# kave=(k1+k2)/2
# klist=kave[-knum:] 
# R=0
# n=30
# sum=np.zeros(n,dtype=complex)
# Rlist=np.arange(n)
# for i in np.arange(n):
#     for kx in klist:
#         sum[i]+=np.exp(-1j*kx*Rlist[i])/knum

# plt.scatter(Rlist,sum.real)
# # plt.plot(Rlist,sum.imag)
# plt.show()



# 创建两个随机的二维数组作为热力图数据
data1 = np.random.rand(10, 10)
data2 = np.random.rand(10, 10)

# 创建一个2x1的子图布局
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(6, 12))

# 在第一个子图上绘制第一个热力图
heatmap1 = ax1.imshow(data1, cmap='hot', interpolation='nearest', aspect='auto')
ax1.set_title('Heatmap 1')
fig.colorbar(heatmap1, ax=ax1, fraction=0.046, pad=0.04)

# 在第二个子图上绘制第二个热力图
heatmap2 = ax2.imshow(data2, cmap='hot', interpolation='nearest', aspect='auto')
ax2.set_title('Heatmap 2')
fig.colorbar(heatmap2, ax=ax2, fraction=0.046, pad=0.04)

# 显示图形
plt.show()
