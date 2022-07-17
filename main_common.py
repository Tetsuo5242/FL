#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import pdb
import matplotlib
matplotlib.use('Agg')#http://t.zoukankan.com/happystudyeveryday-p-13862510.html  在PyCharm中不显示绘图
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
#datasets提供常用的数据集加载，mnist,cifar等
#transforms提供常用的数据预处理操作，主要包括对 Tensor 以及 PIL Image 对象的操作
#tensor是tensorflow的基础概念——张量。包括数据（Data）、流（Flow）、图（Graph）。Tensorflow里的数据用到的都是tensor，所以谷歌起名为tensorflow。
import torch
from utils.sampling import mnist_iid, mnist_noniid, cifar_iid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar
from models.Fed import FedAvg
from models.test import test_img


if __name__ == '__main__':
    # 在该脚本中可以直接运行，若被import到其他脚本中则不会被执行
    # parse args
    args = args_parser()
    # 编写命令行接口，在option.py里有，10个epoch，100个用户
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    #有GPU调用GPU，没有GPU调用CPU

    # 数据下载，以及划分给不同的联邦用户  load dataset and split users
    if args.dataset == 'mnist':
        # option里定义过
        # 将图片transport为tensor类型后进行归一化(normalize)操作。神经网络中归一化可以防止数据溢出，更容易收敛
        #定义了土坡处理方式，下一步调用
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('C:/Users/Lenovo/PycharmProjects/federated-learning-master/data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('C:/Users/Lenovo/PycharmProjects/federated-learning-master/data/mnist/', train=False, download=True, transform=trans_mnist)
        #将数据集下载到data文件夹的mnist和cifar文件夹里

        # sample users 数据分发
        #iid=独立同分布数据=independently identically distribution
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        # noniid=非独立同分布数据
        # 将数据划分为iid和noniid，测试FedAvg在不同场景下的性能
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
            #pdb.set_trace()
        #n 进行下一步
        #对应不同的灰度。 tensor(5)代表第一张图的label，真实值是5
        #pdb是python自带的一个包，交互式的代码调试
        #pdb可以设置断点、单步调试、进入函数调试、查看当前代码、查看zhai片段、动态改变变量的值
        #p dict_users[1] 中的数字代表图片在训练集中的位置。p len(dict_users[1])=600 或 p dict_users[1].shape
        #p args.iid
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            exit('Error: only consider IID setting in CIFAR10')
    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape

    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
        #glob代表全局模型，掌握在服务器手中
    else:
        exit('Error: unrecognized model')
    print(net_glob)
    net_glob.train()
    #不是训练，是切换成训练模式，记录每一批数据的均值和方差，参数更新，反向传播。 对应model.eval()测试模式，不进行记录
    # copy weights
    w_glob = net_glob.state_dict()
    #记录需要学习的weight和bias
    # training，参数的定义
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []

    if args.all_clients:
        print("Aggregation over all clients") # 对所有客户端进行据合，这里并没有真的分发到各服务器，false就行了，进行到下一步。
        w_locals=[w_glob for i in range(args.num_users)]
        """
        另一种写法
        for i in range(args.num_users):
            w_locals.append(w_glob)
        """

    for iter in range(args.epochs):
        #w_locals, loss_locals = [], []
        #w_locals和loss_locals是不同clients手中的
        loss_locals=[]
        if not args.all_clients:
            w_locals=[]
        m = max(int(args.frac * args.num_users), 1)
        #args.frac是options定义的，0.1 num_users是100，每次抽取10个人
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        #上面是模拟随机选取一部分client。全部选择会增加通信量，且实验效果可能不好。
        #google的例子
        #对于每一个被抽到的用户
        for idx in idxs_users:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            #将args,训练集，分到的用户传进去
            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
            w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
            #复制本轮的users的w_locals和loss
        # update global weights
        w_glob = FedAvg(w_locals)
        # 通过定义的FedAvg函数求模型参数的平均

        # copy weight to net_glob
        # state_dict是参数和缓冲区的键值对。load是复制键值对
        net_glob.load_state_dict(w_glob)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_train.append(loss_avg)

        #一个epoch的训练结束，将更新后的模型再分发给每个用户

    # plot loss curve，画图
    plt.figure()
    plt.plot(range(len(loss_train)), loss_train)
    plt.ylabel('train_loss')
    plt.savefig('./save/fed_{}_{}_{}_C{}_iid{}.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid))

    # testing，模型在测试集中测试
    net_glob.eval()
    acc_train, loss_train = test_img(net_glob, dataset_train, args)
    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy: {:.2f}".format(acc_test))

