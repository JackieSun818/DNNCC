from model_train import *
from model_test import *
from model_evaluate import *


# BATCH_SIZE = [128, 256, 512]
# ATT_SIZE = [256, 512, 1024]
# FUN_SIZE = [256, 512, 1024]
# paras = []
# for i in BATCH_SIZE:
#     for j in ATT_SIZE:
#         for k in FUN_SIZE:
#             t = [i, j, k]
#             paras.append(t)
# np.savetxt('tiaocan/paras.txt', np.array(paras), fmt='%d')

paras = np.loadtxt('paras128.txt', dtype=int).tolist()
# paras = [[256, 512, 1024]]
for i in range(len(paras)):
    print('BATCH_SIZE = ', paras[i][0], 'ATT_SIZE = ', paras[i][1], 'FUN_SIZE = ', paras[i][2])
    model_train(paras[i][0], paras[i][1], paras[i][2])
    model_test(paras[i][0], paras[i][1], paras[i][2])
    auc, aupr = model_evaluate(paras[i][0], paras[i][1], paras[i][2])
    file = open('result128.txt', 'a+')
    r = [paras[i][0], paras[i][1], paras[i][2], auc, aupr]
    file.write(str(r))
    file.write('\n')
    file.close()

