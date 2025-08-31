import numpy as np
import matplotlib.pyplot as plt

datapath = "D:/MLDS/"

loss_train_gcn = np.load(datapath + 'loss_train_gcn.npy')
acc_train_gcn = np.load(datapath +'acc_train_gcn.npy')
loss_train_rgcn = np.load(datapath +'loss_train_rgcn.npy')
acc_train_rgcn = np.load(datapath +'acc_train_rgcn.npy')
loss_train_h2g2 = np.load(datapath +'loss_train_h2g2.npy')
acc_train_h2g2 = np.load(datapath +'acc_train_h2g2.npy')

loss_train_gcn_cnn = np.load(datapath +'loss_train_gcn_cnn.npy')
acc_train_gcn_cnn = np.load(datapath +'acc_train_gcn_cnn.npy')
loss_train_rgcn_cnn = np.load(datapath +'loss_train_rgcn_cnn.npy')
acc_train_rgcn_cnn = np.load(datapath +'acc_train_rgcn_cnn.npy')
loss_train_h2g2_cnn = np.load(datapath +'acc_train_h2g2_cnn.npy')
acc_train_h2g2_cnn = np.load(datapath +'acc_train_h2g2_cnn.npy')


loss_val_gcn = np.load(datapath + 'loss_val_gcn.npy')
acc_val_gcn = np.load(datapath +'acc_val_gcn.npy')
loss_val_rgcn = np.load(datapath +'loss_val_rgcn.npy')
acc_val_rgcn = np.load(datapath +'loss_val_rgcn.npy')
loss_val_h2g2 = np.load(datapath +'loss_val_h2g2.npy')
acc_val_h2g2 = np.load(datapath +'loss_val_h2g2.npy')

loss_val_gcn_cnn = np.load(datapath +'loss_val_gcn_cnn.npy')
acc_val_gcn_cnn = np.load(datapath +'acc_val_gcn_cnn.npy')
loss_val_rgcn_cnn = np.load(datapath +'loss_val_rgcn_cnn.npy')
acc_val_rgcn_cnn = np.load(datapath +'loss_val_rgcn_cnn.npy')
loss_val_h2g2_cnn = np.load(datapath +'loss_val_h2g2_cnn.npy')
acc_val_h2g2_cnn = np.load(datapath +'loss_val_h2g2_cnn.npy')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.plot(acc_train_gcn * 100, color = 'blue', label = 'GCN + scalar Features')
ax1.plot(acc_train_gcn_cnn * 100,linestyle='--', color = 'blue', label = 'GCN + CNN Features')
ax1.plot(acc_train_rgcn* 100, color = 'red', label='RGCN + scalar Features')
ax1.plot(acc_train_rgcn_cnn* 100,linestyle='--', color = 'red', label = 'RGCN + CNN Features')
ax1.plot(acc_train_h2g2* 100, color = 'green',label='H2G2-Net + scalar Features')
ax1.plot(acc_train_h2g2_cnn* 100,linestyle='--', color = 'green',label='H2G2-Net + CNN Features')
ax1.set_xlabel('Epochs', fontsize=13)
ax1.set_ylabel('Percentage(%)', fontsize=13)
ax1.set_title('Accuracy of GNNs', fontsize=16)
ax1.set_ylim([20, 85])
ax1.legend()
ax1.plot()

ax2.plot(loss_train_gcn, color = 'blue', label = 'GCN + scalar Features')
ax2.plot(loss_train_gcn_cnn,linestyle='--', color = 'blue', label = 'GCN + CNN Features')
ax2.plot(loss_train_rgcn, color = 'red', label='RGCN + scalar Features')
ax2.plot(loss_train_rgcn_cnn,linestyle='--', color = 'red', label = 'RGCN + CNN Features')
ax2.plot(loss_train_h2g2, color = 'green',label='H2G2-Net + scalar Features')
ax2.plot(loss_train_h2g2_cnn,linestyle='--', color = 'green',label='H2G2-Net + CNN Features')
ax2.set_xlabel('Epochs',fontsize =13)
ax2.set_ylabel('A.U.', fontsize=13)
ax2.set_title('Loss of GNNs',fontsize=16)
ax2.legend()
ax2.plot()
plt.show()


