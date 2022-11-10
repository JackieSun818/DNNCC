import torch.nn as nn  # 神经网络模块
import torch
from my_function import *
import torch.utils.data as Data
import random
import torch.nn.functional as F
# BATCH_SIZE = 2048
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

        self.encoder_fun = nn.Sequential(
            nn.Linear(5603, 4096),
            nn.GELU(),
            nn.Linear(4096, fun_size),
            nn.Sigmoid()
        )

        self.encoder_att = nn.Sequential(
            nn.Linear(size_r, 2048),
            nn.GELU(),
            nn.Linear(2048, att_size),
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


def cosine_sim(e):
    l2_1 = torch.mm(e, e.T)
    l2_2 = torch.mm(torch.sqrt_(torch.sum(e.mul(e), dim=1).view(torch.sum(e.mul(e), dim=1).shape[0], 1)),
                    torch.sqrt_(torch.sum(e.mul(e), dim=1).view(1, torch.sum(e.mul(e), dim=1).shape[0])))
    l2_3 = torch.div(l2_1, l2_2)
    return l2_3


def model_train(BATCH_SIZE, ATT_SIZE, FUN_SIZE):
    # 读取数据
    print('读取数据')

    A = np.loadtxt('../dataset/mat_drug_protein.txt')
    x1 = np.loadtxt('../dataset/RRI.txt')
    x2 = np.loadtxt('../dataset/RPI.txt')
    x3 = np.loadtxt('../dataset/RDI.txt')
    sr = np.loadtxt('../dataset/Similarity_Matrix_Drugs.txt')
    xr_m = sr.dot(np.hstack((x1, x2)))
    for i in range(xr_m.shape[0]):
        for j in range(xr_m.shape[1]):
            if xr_m[i, j] <= 0.5:
                xr_m = 0
    xr_c = x3  # 708*6576

    x4 = np.loadtxt('dataset/PRI.txt')
    x5 = np.loadtxt('dataset/PPI.txt')
    x6 = np.loadtxt('dataset/PDI.txt')
    xp = np.loadtxt('dataset/protein_vector.txt')
    xp = xp / np.max(xp)
    sp = cos_similarity(xp)
    xp_m = sp.dot(np.hstack((x4, x5)))
    for i in range(xp_m.shape[0]):
        for j in range(xp_m.shape[1]):
            if xp_m[i, j] <= 0.5:
                xp_m = 0
    xp_c = x6  # 1512*6576

    index_0 = np.loadtxt('result/DTI_index_0.txt')
    index_1 = np.loadtxt('result/DTI_index_1.txt')

    print('初始化网络')

    for f in range(1):
        # 4份1 共1536个
        fold_index_1 = index_1[0:f, :].flatten().tolist() + index_1[f + 1:FOLD, :].flatten().tolist()
        train_index_0 = index_0[0:f, :].flatten().tolist() + index_0[f + 1:FOLD, :].flatten().tolist()
        num_1 = len(fold_index_1)
        num_0 = len(train_index_0)
        train_index_1 = []
        while num_1 < num_0:
            train_index_1 += fold_index_1
            num_1 = len(train_index_1)
        print(num_1, num_0)
        # train_index_1 += train_index_1
        train_index_1 = train_index_1[0: num_0]
        train_x = []
        train_y = []
        model = MLP_CNN(size_r=xr_m.shape[1], size_p=xp_m.shape[1], att_size=ATT_SIZE, fun_size=FUN_SIZE)
        if CUDA_AVAILABLE == 1:
            model = model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        loss_func1 = nn.CrossEntropyLoss()
        loss_func2 = nn.MSELoss()
        dataset = Data.TensorDataset(torch.from_numpy(np.array(train_index_1)),
                                     torch.from_numpy(np.array(train_index_0)))
        # dataloader = Data.DataLoader(dataset=dataset, batch_size=BATCH_SIZE, num_workers=1, shuffle=True)
        dataloader = Data.DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)
        for epoch in range(EPOCH):
            for step, data in enumerate(dataloader):
                batch_idx_1, batch_idx_0 = data
                # batch_idx = batch_idx_1.numpy().tolist() + batch_idx_0.numpy().tolist()
                # random.shuffle(batch_idx)
                # batch_x_r = []
                batch_idx = torch.cat((batch_idx_1, batch_idx_0), 0)
                batch_y = []
                # batch_x_p = []
                xr_m_tensor = torch.from_numpy(xr_m).float()
                xp_m_tensor = torch.from_numpy(xp_m).float()
                xr_c_tensor = torch.from_numpy(xr_c).float()
                xp_c_tensor = torch.from_numpy(xp_c).float()

                for i in range(len(batch_idx)):
                    drug_no = int(batch_idx[i] / PROTEIN)
                    protein_no = int(batch_idx[i] % PROTEIN)
                    batch_y.append(A[int(drug_no), int(protein_no)])

                batch_idx = torch.from_numpy(np.array(batch_idx)).long()
                batch_y = torch.from_numpy(np.array(batch_y)).long()
                if CUDA_AVAILABLE == 1:
                    batch_idx = batch_idx.cuda()
                    batch_y = batch_y.cuda()
                    xr_m_tensor = xr_m_tensor.cuda()
                    xp_m_tensor = xp_m_tensor.cuda()
                    xr_c_tensor = xr_c_tensor.cuda()
                    xp_c_tensor = xp_c_tensor.cuda()
                er, ep, out, e_cnn = model(xr_m_tensor, xp_m_tensor, xr_c_tensor, xp_c_tensor, batch_idx)

                # cos_r = cosine_sim(er)
                # cos_p = cosine_sim(ep)
                loss1 = loss_func1(out, batch_y)
                loss = loss1
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if step % 10 == 0:
                    if CUDA_AVAILABLE == 1:
                        print('FOLD:', f, 'Epoch: ', epoch, 'Item: ', step, math.ceil(len(train_index_1) / BATCH_SIZE),
                              '| loss: %.20f' % loss1.cpu().item())
                    else:
                        print('FOLD:', f, 'Epoch: ', epoch, 'Item: ', step, ' | ',
                              math.ceil(len(train_index_1) / BATCH_SIZE),
                              '| loss: %.20f' % loss1.item())

        if CUDA_AVAILABLE == 1:
            para = str(BATCH_SIZE) + str(ATT_SIZE) + str(FUN_SIZE)
            torch.save(model.cpu(), 'model/fold_' + str(f) + '_' + str(para) + '.pkl')
        else:
            para = str(BATCH_SIZE) + str(ATT_SIZE) + str(FUN_SIZE)
            torch.save(model, 'model/fold_' + str(f) + '_' + str(para) + '.pkl')
    
