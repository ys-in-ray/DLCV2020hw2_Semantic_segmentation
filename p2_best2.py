# p2.py
import torch
import torch.nn as nn
from torchvision import models
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import imageio
import sys
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
# from pytorch_model_summary import summary
from typing import List,Union,Any

def mean_iou_score(pred, labels):
    '''
    Compute mean IoU score over 6 classes
    '''
    pred = pred.cpu().numpy()
    pred = np.argmax(pred,axis=1)
    labels = labels.cpu().numpy()
    
    N = pred.shape[0]
    
    if mean_iou_score.reset == 1:
        mean_iou_score.tp_fp = [0 for i in range(6)]
        mean_iou_score.tp_fn = [0 for i in range(6)]
        mean_iou_score.tp = [0 for i in range(6)]
        mean_iou_score.reset = 0

    iou = [0 for i in range(6)]
    mean_iou = 0
    for i in range(6):
        for j in range(N):
            mean_iou_score.tp_fp[i] += np.sum(pred[j,:,:] == i)
            mean_iou_score.tp_fn[i] += np.sum(labels[j,:,:] == i)
            mean_iou_score.tp[i] += np.sum((pred[j,:,:] == i) * (labels[j,:,:] == i))
    for i in range(6):
        iou[i] = mean_iou_score.tp[i] / (mean_iou_score.tp_fp[i] + mean_iou_score.tp_fn[i] - mean_iou_score.tp[i])
        mean_iou += iou[i] / 6
        # print('class #%d : %1.5f, tp_fp: %d, tp_fn: %d, tp: %d,'%(i, iou[i], mean_iou_score.tp_fp[i], mean_iou_score.tp_fn[i], mean_iou_score.tp[i]))
    # print('\nmean_iou: %f\n' % mean_iou)
    return mean_iou

def check_result(model,train_set,valid_set,cost_func,device = torch.device("cuda"),batch_size =8):
    times = 257//batch_size
    total_valid_miou = 0.0
    total_valid_loss = 0.0
    total_train_miou = 0.0
    total_train_loss = 0.0
    model.eval()
    with torch.no_grad():
        valid_loader = DataLoader(dataset=valid_set, batch_size=batch_size, shuffle=True)
        mean_iou_score.reset =1
        for i ,data in enumerate(valid_loader,0):
            inputs , mylabel = data

            inputs = inputs.to(device)
            mylabel = mylabel.to(device)
            outputs = model(inputs)
            myout = torch.argmax(outputs,dim=1)

            valid_miou = mean_iou_score(outputs, mylabel)
            valid_loss = cost_func(outputs, mylabel)

            total_valid_loss+=valid_loss
        total_valid_loss /=i
        #train set per epoch
        train_check_loader = DataLoader(dataset=train_set,batch_size = batch_size,shuffle=True)
        mean_iou_score.reset =1
        for i in range(times):
            train_inputs , my_train_label = next(iter(train_check_loader))

            train_inputs = train_inputs.to(device)
            my_train_label = my_train_label.to(device)
            train_outputs = model(train_inputs)
            my_train_out = torch.argmax(train_outputs,dim=1)

            train_miou = mean_iou_score(train_outputs, my_train_label)
            train_loss = cost_func(train_outputs, my_train_label)
            total_train_loss += train_loss
        total_train_loss/=i

    print('train[miou=%.4f,loss=%.4f], valid[miou=%.4f,loss=%.4f]'%(train_miou,total_train_loss,valid_miou,total_valid_loss))
    model.train()
    return valid_miou,total_valid_loss

class P2_Dataset(Dataset):
    def __init__(self,path,len=2000):
        # read file
        self.path = path
        self.len = len

    def __getitem__(self, idx):
        sat_filename = self.path+str(idx).zfill(4)+'_sat.jpg'
        im = Image.open(sat_filename)
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]

        # trans_list =[]
        trans = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean, std)
                                    ])
        im = trans(im)

        mask_filename = self.path+str(idx).zfill(4)+'_mask.png'
        mask = Image.open(mask_filename)
        mask = transforms.ToTensor()(mask)
        mask = mask==0
        a = (mask[0,:,:]*4+mask[1,:,:]*2+mask[2,:,:])
        a_shape =a.shape
        a = a.view(-1)
        convert = torch.tensor([5,1,2,6,0,3,4,6])
        b = torch.gather(convert,0,a).view(a_shape)

        return im,b
    def __len__(self):
        return self.len

class FCN32(nn.Module):
    def __init__(self,features: nn.Module):
        super().__init__()
        self.features = features
        # feature_list = list(self.features.children())
        # print(len(feature_list))
        for i in range (17):
            for param in self.features[i].parameters():
                param.requires_grad = False
        # self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.conv=nn.Sequential(nn.Conv2d(512,512,kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                nn.ReLU(inplace=True),
                                nn.Dropout2d(),
                                nn.Conv2d(512,512,kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                nn.ReLU(inplace=True),
                                nn.Dropout2d(),
                                )
        self.upsample = nn.Sequential(nn.ConvTranspose2d(1024,256,kernel_size=3, stride=2, padding=1,output_padding=1),
                                    nn.ReLU(inplace=True),
                                    nn.ConvTranspose2d(768,128,kernel_size=3, stride=2, padding=1,output_padding=1),
                                    nn.ReLU(inplace=True),
                                    nn.ConvTranspose2d(384,64,kernel_size=3, stride=2, padding=1,output_padding=1),
                                    nn.ReLU(inplace=True),
                                    nn.ConvTranspose2d(64,32,kernel_size=3, stride=2, padding=1,output_padding=1),
                                    nn.ReLU(inplace=True),
                                    nn.ConvTranspose2d(32,7,kernel_size=3, stride=2, padding=1,output_padding=1),
                                    nn.ReLU(inplace=True)
                                    )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(17):
            x = self.features[i](x)
        x1 = x
        for i in range(17,24):
            x = self.features[i](x)
        x2 = x
        for i in range(24,31):
            x = self.features[i](x)
        x3 = x

        x = self.conv(x)
        x = torch.cat((x,x3),dim=1)
        for i in range(2):
            x = self.upsample[i](x)
        x = torch.cat((x,x2),dim=1)
        for i in range(2,4):
            x = self.upsample[i](x)
        x = torch.cat((x,x1),dim=1)
        for i in range(4,10):
            x = self.upsample[i](x)

        return x

def save_best_model(model,miou,loss,epoch,version,doit = False):
    if save_best_model.reset == 1:
        save_best_model.bmiou = 0.0
        save_best_model.bloss = 100.0
        save_best_model.reset = 0

    if doit:
        torch.save(model.state_dict(),'p2_model/'+version+'e'+str(epoch)+'.pth')


    if miou >= save_best_model.bmiou and loss <= save_best_model.bloss:
        torch.save(model.state_dict(),'p2_model/'+version+'.pth')
        save_best_model.bmiou = miou
        save_best_model.bloss = loss
        print('===save![%d]==='%(epoch))
    elif miou > save_best_model.bmiou:
        torch.save(model.state_dict(),'p2_model/'+version+'_m.pth')
        save_best_model.bmiou = miou
        print('===save!miou[%d]==='%(epoch))
    elif loss < save_best_model.bloss:
        torch.save(model.state_dict(),'p2_model/'+version+'_l.pth')
        save_best_model.bloss = loss
        print('===save!loss[%d]==='%(epoch))

def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            # v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

def _vgg( batch_norm: bool ,**kwargs: Any) -> FCN32:
    cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
    model = FCN32(make_layers(cfg, batch_norm=batch_norm), **kwargs)
    # model.load_state_dict(state_dict)
    return model


if __name__ == '__main__':
    version = 'p2_best'
    n_epochs = 150
    batch_size = 8
    learning_rate = 1e-4

    #init
    save_best_model.reset = 1

    model = _vgg(batch_norm=False)

    pretrained_model = models.vgg16(pretrained=True)
    pretrained_dict = pretrained_model.state_dict()
    model_dict = model.state_dict()
    # print(*model_dict.keys())
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if "classifier." not in k}
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict) 
    # print(*model_dict.keys())
    model.load_state_dict(model_dict)

    print(summary(model,torch.zeros((5, 3, 512, 512)), show_input=True))

    if torch.cuda.is_available():
        model = model.cuda()
        print('use cuda')

    train_set = P2_Dataset(path ='hw2_data/p2_data/train/',len=2000)
    valid_set = P2_Dataset(path ='hw2_data/p2_data/validation/',len=257)

    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
    cost = nn.CrossEntropyLoss()

    for epoch in range(n_epochs):
        if epoch%3 ==0:
            if epoch == 3 or epoch ==12 or epoch == 20:
                m,l =check_result(model,train_set,valid_set,cost)
                save_best_model(model,m,l,epoch,version,doit = True)
            else:
                m,l =check_result(model,train_set,valid_set,cost)
                save_best_model(model,m,l,epoch,version,doit = False)
        else:
            print()

        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            # inputs =  np.transpose(inputs, (0, 3, 1, 2)) # batch*32*32*3 -> batch*3*32*32
            device = torch.device("cuda")
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            optimizer.zero_grad()    
            loss = cost(outputs, labels)
            loss.backward()
            optimizer.step()

            percentage = i*batch_size/(len(train_set))*100
            sys.stdout.write('\r')
            sys.stdout.write("ep[%3d] "%(epoch))
            sys.stdout.write("[%-20s] %d%%" % ('='*int(percentage/5), percentage+1))
            # sys.stdout.write(" %.4f" % (loss))
            print

    check_result(model,train_set,valid_set,cost)


