import torch.nn as nn  # 神经网络模块
import torch
from my_function import *
import torch.utils.data as Data
import random
import torch.nn.functional as F
BATCH_SIZE = 1024
EPOCH = 100
LR = 0.0001
FOLD = 5
CUDA_AVAILABLE = 1
DRUG = 708
PROTEIN = 1512


class MLP_CNN(nn.Module):
    def __init__(self, size_r, size_p, att_size, fun_size):
        super(MLP_CNN, self).__init__()
        self.conv_l1 = nn.Sequential(  # input shape (1,2,256)
            nn.Conv2d(in_channels=1,  # input height
                      out_channels=16,  # n_filter
                      kernel_size=(3, 5),  # 6*260->4*256
                      padding=2  # con2d出来的图片大小不变   2*256->6*260
                      ),  # output shape [1, 16, 4, 256]
            nn.LeakyReLU(),
            # nn.MaxPool2d(2),  # [1, 1, 4, 256] -> [1, 1, 2, 128]
            nn.AvgPool2d(2)
        )

        self.conv_l2 = nn.Sequential(
            nn.Conv2d(in_channels=16,  # input height
                      out_channels=32,  # n_filter
                      kernel_size=(3, 5),  # filter size
                      padding=2  # con2d出来的图片大小不变
                      ),  # output shape [1, 32, 4, 128]
            nn.LeakyReLU(),
            nn.MaxPool2d(2),  # [1, 32, 4, 128] -> [1, 32, 2, 64]
        )
        self.out = nn.Linear(int((att_size + fun_size) / 4 * 2 * 32), 2)

        self.encoder_protein = nn.Sequential(
            nn.Linear(size_p, size_r),
            # nn.Dropout(0.2),
            nn.Sigmoid()
        )

        self.encoder_fun = nn.Sequential(
            nn.Linear(6576, 4096),
            # nn.Dropout(0.2),  # 参数是扔掉的比例
            nn.GELU(),
            # nn.Linear(4096, 1024),
            # nn.Dropout(0.2),
            # nn.GELU(),
            nn.Linear(4096, fun_size),
            # nn.Dropout(0.2),
            nn.Sigmoid()
        )

        self.encoder_att = nn.Sequential(
            nn.Linear(size_r, 2048),
            # nn.Dropout(0.2),  # 参数是扔掉的比例
            nn.GELU(),
            # nn.Linear(2048, 512),
            # nn.Dropout(0.2),
            # nn.GELU(),
            nn.Linear(2048, att_size),
            # nn.Dropout(0.2),
            nn.Sigmoid()
        )

    def forward(self, r_att, p_att, r_fun, p_fun, idx):
        # e_pro_att = self.encoder_protein(p_att)
        att = torch.cat((r_att, p_att), 0)
        e_att = self.encoder_att(att)

        fun = torch.cat((r_fun, p_fun), 0)
        e_fun = self.encoder_fun(fun)
        e = torch.cat((e_att, e_fun), 1)
        e_r = e[0: r_fun.shape[0], :]
        e_p = e[r_fun.shape[0]:, :]

        x_cnn = []
        for i in range(idx.shape[0]):
            r_no = int(idx[i] / PROTEIN)
            p_no = int(idx[i] % PROTEIN)
            x_cnn.append(torch.cat((e_r[r_no, :], e_p[p_no, :]), 0))
        x_cnn = torch.cat(tuple(x_cnn), 0)
        if CUDA_AVAILABLE == 1:
            x_cnn = x_cnn.cuda()

        x_cnn = x_cnn.view(-1, 1, 2, e.shape[1])
        embedding_cnn = self.conv_l1(x_cnn)  # [1, 1, 2, 256] -> [1, 16, 2, 128]
        embedding_cnn = self.conv_l2(embedding_cnn)  # [1, 16, 2, 128] -> [1, 32, 2, 64]
        embedding_cnn = torch.Tensor.tanh(embedding_cnn)
        b, n_f, h, w = embedding_cnn.shape
        output = self.out(embedding_cnn.view(b, n_f * h * w))
        return e_r, e_p, output, embedding_cnn.view(b, n_f * h * w)


def model_test(BATCH_SIZE, ATT_SIZE, FUN_SIZE):
    # 读取数据
    print('读取数据')
    para = str(BATCH_SIZE) + str(ATT_SIZE) + str(FUN_SIZE)
    A = np.loadtxt('../dataset/mat_drug_protein.txt')
    x1 = np.loadtxt('../dataset/RRI.txt')
    x2 = np.loadtxt('../dataset/RPI.txt')
    x3 = np.loadtxt('../dataset/RDI.txt')
    sr = np.loadtxt('../dataset/Similarity_Matrix_Drugs.txt')
    xr_m = sr.dot(np.hstack((x1, x2)))
    xr_c = x3  # 708*6576

    x4 = np.loadtxt('../dataset/PRI.txt')
    x5 = np.loadtxt('../dataset/PPI.txt')
    x6 = np.loadtxt('../dataset/PDI.txt')
    sp = np.loadtxt('../dataset/Similarity_Matrix_Proteins.txt')
    xp_m = sp.dot(np.hstack((x4, x5)))
    xp_c = x6  # 1512*6576

    index_0 = np.loadtxt('../result/DTI_index_0.txt')
    index_1 = np.loadtxt('../result/DTI_index_1.txt')

    print('初始化网络')
    for f in range(1):

        # 预测测试集中的0
        test_index = index_1[f, :].flatten().tolist() + index_0[f, :].flatten().tolist()
        model = torch.load('model/fold_' + str(f) + '_' + str(para) + '.pkl')
        if CUDA_AVAILABLE == 1:
            model = model.cuda()

        dataset = Data.TensorDataset(torch.from_numpy(np.array(test_index)))
        dataloader = Data.DataLoader(dataset=dataset, batch_size=2*BATCH_SIZE, shuffle=False)
        fold_out = []
        fold_out1 = []
        fold_index = []
        for step, data in enumerate(dataloader):
            batch_idx = data[0]
            xr_m_tensor = torch.from_numpy(xr_m).float()
            xp_m_tensor = torch.from_numpy(xp_m).float()
            xr_c_tensor = torch.from_numpy(xr_c).float()
            xp_c_tensor = torch.from_numpy(xp_c).float()

            if CUDA_AVAILABLE == 1:
                batch_idx = batch_idx.cuda()
                # batch_y = batch_y.cuda()
                xr_m_tensor = xr_m_tensor.cuda()
                xp_m_tensor = xp_m_tensor.cuda()
                xr_c_tensor = xr_c_tensor.cuda()
                xp_c_tensor = xp_c_tensor.cuda()
            er, ep, batch_out, e_cnn = model(xr_m_tensor, xp_m_tensor, xr_c_tensor, xp_c_tensor, batch_idx)

            if CUDA_AVAILABLE == 1:
                fold_out += batch_out.cpu().detach().numpy().tolist()
                fold_index += batch_idx.cpu().detach().numpy().tolist()
                # print(len(fold_out))
                # er_list = er.cpu().detach().numpy().tolist()
                # ep_list = ep.cpu().detach().numpy().tolist()

            else:
                fold_out += batch_out.detach().numpy().tolist()
                fold_index += batch_idx.detach().numpy().tolist()
                # er_list = er.detach().numpy().tolist()
                # ep_list = ep.detach().numpy().tolist()
            print('FOLD: ', f, 'Item: ', step, math.ceil(len(test_index) / (2 * BATCH_SIZE)))

        np.savetxt('result/pre_fold_' + str(f) + '_' + str(para) + '.txt', np.array(fold_out), fmt='%.6f')
        np.savetxt('result/idx_fold_' + str(f) + '_' + str(para) + '.txt', np.array(fold_index), fmt='%.6f')
        # np.savetxt('result/DNNCE药物编码结果.txt', np.array(er_list), fmt='%.6f')
        # np.savetxt('result/DNNCE蛋白编码结果.txt', np.array(ep_list), fmt='%.6f')


