import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append("..")
import crypten
import crypten.mpc as mpc
import crypten.communicator as comm

from crypten.mpc import MPCTensor
from crypten.mpc.ptype import ptype as Ptype

import torchvision
import torchvision.models as models
import torch.autograd.profiler as profiler

import logging
import time
import timeit
import argparse

from network import *
import torchvision
from torch.utils.data import DataLoader
def test_accuracy(model, input_size, batch_size, num_classes, device="cuda"):
    comm.get().set_verbosity(True)
    rank = comm.get().get_rank()
    #读取数据
    #train_data = torchvision.datasets.MNIST(root='./mnist/', train=True, transform=torchvision.transforms.ToTensor(), download=True)
    #test_data = torchvision.datasets.MNIST(root='./mnist/', train=False, transform=torchvision.transforms.ToTensor(), download=True)
    train_data = torchvision.datasets.CIFAR10(root='./cifar10/', train=True, transform=torchvision.transforms.ToTensor(), download=True)
    
    test_data = torchvision.datasets.CIFAR10(root='./cifar10/', train=False, transform=torchvision.transforms.ToTensor(), download=True)
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    c, h, w = input_size
    bs = batch_size
    #print(model)
    criterion = crypten.nn.CrossEntropyLoss()
    model = crypten.nn.from_pytorch(model, dummy_input=torch.empty(bs, c, h, w))
    #print(model._modules)
    model = model.to(device)
    model.encrypt() 
    model.train()

    for data in train_dataloader:
        #it_num += 1
        img, targets = data
        img = img.cuda()
        targets = targets.cuda()

        labels = F.one_hot(targets, num_classes=num_classes)

        labels = crypten.cryptensor(labels, src=0)
        #input = torch.randn([bs,c,w,h], requires_grad=False)
        input = crypten.cryptensor(img, src=0)
        #print(labels)
        #训练
        tic = time.perf_counter()
        output = model(input)
        toc = time.perf_counter()
        #print(output.get_plain_text())
        loss = criterion(output, labels)
        print(loss.get_plain_text(),toc-tic)

        model.zero_grad()
        loss.backward()
        model.update_parameters(learning_rate=0.1)
        
    
    #预测部分
    model.eval()
    #model.replicate_parameters()
    total_accuracy = 0
    total_num = 0 
    #tic = time.perf_counter()
    ii_num=0
    
    for data in test_dataloader:
        ii_num+=1
        img, targets = data
        img = img.cuda()
        targets = targets.cuda()
        
        labels = F.one_hot(targets, num_classes=num_classes)
        labels = crypten.cryptensor(labels, src=0)

        input = crypten.cryptensor(img, src=0).to(device)
        
        output = model(input)
        
        #loss = criterion(output, labels)

        output = output.get_plain_text()
        #print(output.argmax(1))

        total_accuracy += (output.argmax(1) == targets).sum()
        
        total_num += targets.shape[0]
        print(total_accuracy,total_num)
   
    #print(f"infer Runtime: {(t_t) / ii_num}")
    print(f"accuracy: {total_accuracy / total_num}")
def inference(model, input_size, batch_size=1, device="cuda"):
    comm.get().set_verbosity(True)

    bs = batch_size
    c, w, h = input_size
    x = crypten.cryptensor(torch.rand((bs, c, w, h)), device=device, requires_grad=False)

    model = crypten.nn.from_pytorch(model, dummy_input=torch.empty(bs, c, w, h))
    model = model.encrypt()
    model = model.to(device) 

    model.eval()
    model.replicate_parameters()

    total_time = 0
    comm_time = 0
    conv_time, pool_time, relu_time, matmul_time = 0, 0, 0, 0
    for i in range(6):
        comm.get().reset_communication_stats()
        
        tic = time.perf_counter()
        model(x)
        toc = time.perf_counter()

        if i != 0:
            total_time += toc - tic
            comm_time += comm.get().comm_time
            conv_time += comm.get().time_conv
            relu_time += comm.get().time_relu
            pool_time += comm.get().time_pool
            matmul_time += comm.get().time_matmul

            # if comm.get().get_rank() == 0:
            #     print(f"Iteration {i} runtime: {toc - tic}")

        comm.get().print_total_communication()

    if comm.get().get_rank() == 0:
        print("----------- Statistics ----------------")
        print(f"Total Communication: {comm.get().total_comm_bytes}")
        print(f"Avg Runtime: {total_time / 5}")
        print(f"Avg Comm: {comm_time / 5}")
        print(f"Avg Linear: {(conv_time + matmul_time)/ 5}")
        print(f"Avg ReLU: {relu_time / 5}")
        print(f"Avg Pool: {pool_time / 5}")


def training(model, input_size, batch_size, num_classes, device="cuda"):
    comm.get().set_verbosity(True)
    rank = comm.get().get_rank()

    c, h, w = input_size
    bs = batch_size
    criterion = crypten.nn.CrossEntropyLoss()
    model = crypten.nn.from_pytorch(model, dummy_input=torch.empty(bs, c, h, w))
    model = model.to(device)
    model.encrypt() 
    model.train()

    labels = torch.ones(bs, requires_grad=False).long().to(device)
    labels = F.one_hot(labels, num_classes=num_classes)
    labels = crypten.cryptensor(labels, src=0)

    input = torch.randn([bs,c,w,h], requires_grad=False)
    input = crypten.cryptensor(input, src=0).to(device)

    total_time = 0
    comm_time = 0
    conv_time, pool_time, relu_time, matmul_time, softmax_time = 0, 0, 0, 0, 0
    #from torch.cuda.amp import autocast as autocast
    time_t=6
    for i in range(time_t):
        comm.get().reset_communication_stats()
        tic = time.perf_counter()
        #with autocast():
        output = model(input)

        loss = criterion(output, labels)

        #model.zero_grad()
        loss.backward()
        model.update_parameters(learning_rate=0.1)

        toc = time.perf_counter()

        if i != 0:
            total_time += toc - tic
            comm_time += comm.get().comm_time
            conv_time += comm.get().time_conv
            relu_time += comm.get().time_relu
            pool_time += comm.get().time_pool
            matmul_time += comm.get().time_matmul
            softmax_time += comm.get().time_softmax

            # if comm.get().get_rank() == 0:
            #     print(f"Iteration {i} runtime: {toc - tic}")

        comm.get().print_total_communication()

    if comm.get().get_rank() == 0:
        print("----------- Statistics ----------------")
        print(f"Total Communication: {comm.get().total_comm_bytes}")
        print(f"Avg Runtime: {total_time / 5}")
        print(f"Avg Comm: {comm_time / 5}")
        print(f"Avg Linear: {(conv_time + matmul_time )/ 5}")
        print(f"Avg ReLU: {relu_time / 5}")
        print(f"Avg Pool: {pool_time / 5}")
        print(f"Avg Softmax: {softmax_time / 5}")



def inference_plaintext(model, input_size, device="cuda"):

    c, w, h = input_size
    x = torch.rand((1, c, w, h), device=device, requires_grad=False)

    model = model.to(device) 
    model.eval()

    total_time = 0
    for i in range(101):
        comm.get().reset_communication_stats()
        
        tic = time.perf_counter()
        model(x)
        toc = time.perf_counter()

        if i != 0:
            total_time += toc - tic

        comm.get().print_total_communication()

    if comm.get().get_rank() == 0:
        print("----------- Statistics ----------------")
        print(f"Avg Runtime: {total_time / 100}")


def training_plaintext(model, input_size, batch_size, num_classes, device="cuda"):

    c, h, w = input_size
    bs = batch_size

    model = model.to(device)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


    input = torch.randn([bs,c,w,h], requires_grad=True).to(device)
    labels =  torch.ones(bs, requires_grad=False).long().to(device)

    total_time = 0
    comm_time = 0
    conv_time, pool_time, relu_time, matmul_time = 0, 0, 0, 0
    for i in range(101):
        tic = time.perf_counter()

        output = model(input)
        optimizer.zero_grad()
        
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        toc = time.perf_counter()

        if i != 0:
            total_time += toc - tic

            # if comm.get().get_rank() == 0:
            #     print(f"Iteration {i} runtime: {toc - tic}")

    if comm.get().get_rank() == 0:
        print("----------- Statistics ----------------")
        print(f"Avg Runtime: {total_time / 100}")


def select_model(dataset, network):
    if dataset == "mnist":
        input_size = (1,28,28)
        num_classes = 10
        if network == "lenet":
            model = LeNet()
    elif dataset == "cifar10":
        input_size = (3,32,32)
        num_classes = 10
        if network == "alexnet":
            model = AlexNet(num_classes=10)
        elif network == "vgg16":
            model = VGG16(num_classes=10)
    elif dataset == 'tinyin':
        input_size = (3,64,64)
        num_classes = 200
        if network == 'alexnet':
            model = AlexNet(num_classes=200)
        elif network == "vgg16":
            model = VGG16(num_classes=200)
    elif dataset == 'imagenet':
        input_size = (3, 224, 224)
        num_classes = 1000
        if network == 'alexnet':
            model = AlexNet(num_classes=1000)
        elif network == "vgg16":
            model = VGG16(num_classes=1000)
        elif network == "resnet34":
            model = models.resnet34()
            model.maxpool = nn.AvgPool2d(kernel_size=3, stride=2)
        elif network == "resnet50":
            model = models.xin()
            model.maxpool = nn.AvgPool2d(kernel_size=3, stride=2)
        elif network == "resnet101":
            model = models.resnet101()
            model.maxpool = nn.AvgPool2d(kernel_size=3, stride=2)
        elif network == "resnet152":
            model = models.resnet152()
            model.maxpool = nn.AvgPool2d(kernel_size=3, stride=2)

    return model, input_size, num_classes

def train_all():
    train_config = [
        ["mnist", "lenet", 128],
        ["cifar10", "alexnet", 128],
        ["cifar10", "vgg16", 32],
        ["tinyin", "alexnet", 128],
        ["tinyin", "vgg16", 8],
    ]
    for dataset, network, bs in train_config:
        model, input_size, num_classes = select_model(dataset, network)
        if comm.get().get_rank() == 0:
            print(f"Training on {dataset} dataset with {network} network")
        training(model, input_size, bs, num_classes, device="cuda")


def inference_all():
    inference_config = [
        ["mnist", "lenet"],
        ["cifar10", "alexnet"],
        ["cifar10", "vgg16"],
        ["tinyin", "alexnet"],
        ["tinyin", "vgg16"],
        ["imagenet", "alexnet"],
        ["imagenet", "vgg16"],
        ["imagenet", "resnet50"],
        ["imagenet", "resnet101"],
        ["imagenet", "resnet152"]
    ]
    for dataset, network in inference_config:
        model, input_size, num_classes = select_model(dataset, network)
        if comm.get().get_rank() == 0:
            print(f"Running inference on {dataset} dataset with {network} network")
        inference(model, input_size, device="cuda")


def train_all_plaintext():
    train_config = [
        ["mnist", "lenet", 128],
        ["cifar10", "alexnet", 128],
        ["cifar10", "vgg16", 32],
        ["tinyin", "alexnet", 128],
        ["tinyin", "vgg16", 8],
    ]
    for dataset, network, bs in train_config:
        model, input_size, num_classes = select_model(dataset, network)
        if comm.get().get_rank() == 0:
            print(f"Training on {dataset} dataset with {network} network")
        training_plaintext(model, input_size, bs, num_classes, device="cuda")


def inference_all_plaintext():
    inference_config = [
        ["mnist", "lenet"],
        ["cifar10", "alexnet"],
        ["cifar10", "vgg16"],
        ["tinyin", "alexnet"],
        ["tinyin", "vgg16"],
        ["imagenet", "alexnet"],
        ["imagenet", "vgg16"],
        ["imagenet", "resnet50"],
        ["imagenet", "resnet101"],
        ["imagenet", "resnet152"]
    ]
    for dataset, network in inference_config:
        model, input_size, num_classes = select_model(dataset, network)
        if comm.get().get_rank() == 0:
            print(f"Running inference on {dataset} dataset with {network} network")
        inference_plaintext(model, input_size, device="cuda")


def batch_inference():
    inference_config = [
        ["cifar10", "alexnet", 64],
        ["cifar10", "vgg16", 64],
        ["imagenet", "resnet50", 8],
        ["imagenet", "resnet101", 8],
        ["imagenet", "resnet152", 8]
    ]

    for dataset, network, bs in inference_config:
        model, input_size, num_classes = select_model(dataset, network)
        inference(model, input_size, bs, device='cuda')

# A playground to test different network and dataset combinations
def test():
    dataset = "mnist"

    network = "lenet"
    device = "cuda"
    train = False
    batch_size = 128
    
    model, input_size, num_classes = select_model(dataset, network)


    if train:
        training(model, input_size, batch_size, num_classes, device)
    else:
        inference(model, input_size, batch_size, device)

def acc():
    dataset = "cifar10"

    network = "alexnet"
    device = "cuda"
    train = True
    batch_size = 128
    
    model, input_size, num_classes = select_model(dataset, network)

    test_accuracy(model, input_size, batch_size, num_classes, device)

parser = argparse.ArgumentParser()
experiments = ['test', 'train_all', 'inference_all', 'train_all_plaintext', 'inference_all_plaintext', 'batch_inference','acc','training_plaintext']
parser.add_argument(
    "--exp",
    "-e",
    required=False,
    default="test",
    help="Experiment to run",
)

if __name__ == '__main__':
    #测试relu
    '''
    import re_lu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a=torch.randint(1,2,[1500])
    b=torch.randint(1,1000,[1500])
    c=torch.randint(1,10,[1500])

    a=a.to(device)
    b=b.to(device)
    c=c.to(device)

    tic = time.perf_counter()
    re_lu.torch_launch_relu(c,a,b,b,1500)
    print(c)
    toc = time.perf_counter()
    print(toc-tic)
    
    import mat_mul
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a=torch.randint(-10000,10000,[128,500])
    b=torch.randint(-100000,1000000,[500,20])
    c=torch.randint(1,2,[128,20])

    a=a.to(device)
    b=b.to(device)
    c=c.to(device)
    import crypten
    crypten.init()
    x=crypten.cuda.CUDALongTensor(a)
    y=crypten.cuda.CUDALongTensor(b)                                                                                                                                             
    tic = time.perf_counter()
    z=crypten.cuda.CUDALongTensor.matmul(x, y)
    toc = time.perf_counter()
    print(toc-tic)
    print(z)
    tic = time.perf_counter()
    mat_mul.torch_launch_matmul(c,a,b,128, 500, 20)
    print(c)
    toc = time.perf_counter()
    print(toc-tic)

    import mat_mul
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a = torch.LongTensor([(1<<62)-1])
    b = torch.LongTensor([-((1<<62)-2)])

    c=a*1024*1024
    d=b*1024*1024
    print(c+d)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a=torch.randint(0,10,[2,3,3,3],device=device)
    print(a)
    b = a.permute(1,0,2,3)
    print(b.reshape(b.shape[0],-1))
    
    
    import torchvision
    from torch.utils.data import DataLoader

    train_data = torchvision.datasets.MNIST(
        root='./mnist/',
        train=True,                                     # this is training data
        transform=torchvision.transforms.ToTensor(),    # Converts a PIL.Image or numpy.ndarray to
                                                        # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
        download=True,
    )
    train_dataloader = DataLoader(train_data, batch_size=10)
    for data in train_dataloader:
        img, targets=data
        print(img.shape,targets.shape)
    
    
    from crypten.common.util import im2col_indices, col2im_indices, im2col
    input = torch.randint(0,10,[2,4])
    bias = torch.randint(1,2,[2])
    print(input)
    print(input.transpose(0,1)-bias)
    #input = input.reshape(1,-1)
    #a = im2col(input, 2, 2, padding=0, stride=2)
    #a = a.reshape(2,2,-1)
    #print(a)
    '''
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    import multiprocess_launcher
    args = parser.parse_args()
    assert args.exp in experiments
    func = globals()[args.exp]

    launcher = multiprocess_launcher.MultiProcessLauncher(
        3, func,
    )
    launcher.start()
    launcher.join()
    launcher.terminate()
    
