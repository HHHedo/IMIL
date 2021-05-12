#!/usr/bin/env
# -*- coding: utf-8 -*-
'''
Copyright (c) 2019 gyfastas
'''
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch.utils.model_zoo as model_zoo
import math
from .backbones.ResNet import ResNet10, ResNet18, ResNet34, ResNet50
import faiss
class BaseClsNet(nn.Module):
    '''
    Basic classification network. With single classification head and single backbone.

    '''
    def __init__(self, backbone, num_classes=2):
        super(BaseClsNet, self).__init__()
        # self.backbone = backbone
        # self.avgpool = nn.AdaptiveAvgPool2d(1)
        # self.classifier = nn.Sequential(
        #     nn.Linear(backbone.feat_dim, 500),
        #     nn.ReLU(),
        #     nn.Linear(500, num_classes))
        self.classifier = nn.Linear(backbone.feat_dim, num_classes)
        # for m in self.modules():
            # if isinstance(m, nn.Conv2d):
            #     n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            #     m.weight.data.normal_(0, math.sqrt(2. / n))
            # elif isinstance(m, nn.BatchNorm2d):
            #     m.weight.data.fill_(1)
            #     m.bias.data.zero_()
            # if isinstance(m, nn.Linear):
            #     m.weight.data.normal_(mean=0.0, std=0.01)
            #     m.bias.data.zero_()


    def forward(self, x):
        # x = self.backbone(x)
        # x = self.avgpool(x)
        x = self.classifier(x.view(x.shape[0],-1))

        return x

# , num_classes=2 ,  backbone.feat_dim
# prior uniform dis; dic ;mb
class CausalPredictor(nn.Module):
    def __init__(self, backbone, dic, num_classes):
        super(CausalPredictor, self).__init__()

        num_classes = num_classes
        self.embedding_size = 512
        representation_size = backbone.feat_dim

        self.causal_score = nn.Linear(2*representation_size, num_classes)
        self.cluster_score = nn.Linear(representation_size, 10)
        self.Wy = nn.Linear(representation_size, self.embedding_size)
        self.Wz = nn.Linear(representation_size, self.embedding_size)

        nn.init.normal_(self.causal_score.weight, std=0.01)
        nn.init.normal_(self.cluster_score.weight, std=0.01)
        nn.init.normal_(self.Wy.weight, std=0.02)
        nn.init.normal_(self.Wz.weight, std=0.02)
        # self.Wy.weight.data.copy_(torch.eye(representation_size))
        # self.Wz.weight.data.copy_(torch.eye(representation_size))
        nn.init.constant_(self.Wy.bias, 0)
        nn.init.constant_(self.Wz.bias, 0)
        nn.init.constant_(self.causal_score.bias, 0)
        nn.init.constant_(self.cluster_score.bias, 0)

        self.feature_size = representation_size
        self.dic = self.get_conf(dic)
        self.prior = torch.tensor(np.ones(self.dic.shape[0]) / (self.dic.shape[0]),
                                  dtype=torch.float)
        self.dropout = nn.Dropout(p=0.5)
        self.cos  = nn.CosineSimilarity(dim=1, eps=1e-6)
        # self.old_backbone = backbone.eval()
        # self.prior = torch.tensor(np.ones(int(dic[:, -2:-1].max().item()+1))/(dic[:, -2:-1].max().item()+1), dtype=torch.float)
        # print(self.dic.shape, self.prior)

    def get_conf(self, dic):
        
        
        # feature = dic[:, :-2]
        # bag_num = dic[:, -2:-1].max()
        # conf_list = [feature[(dic[:, -2:-1]==num).squeeze(1)].mean(axis=0) for num in range(int(bag_num+1))]
        # return torch.stack(conf_list)
        print('clustering for confounder') 
        feature = dic[:, :-2].numpy().astype('float32')
        bag_num = dic[:, -2].max()
        ncentroids = 10
        niter = 20
        verbose = True
        d = feature .shape[1]
        kmeans = faiss.Kmeans(d, ncentroids, niter=niter, verbose=verbose)
        kmeans.cp.max_points_per_centroid =100000000
        kmeans.train(feature)
        self.kmeans = kmeans
        self.feature = feature 
        self.bag_idx = dic[:, -2].numpy().astype('float32')
        self.inner_idx = dic[:, -1].numpy().astype('float32')
        return torch.from_numpy(kmeans.centroids)

    def trunk_attention(self, a, dic_z):
        with torch.no_grad():
            a = F.softmax(a, 1)
            sorted_a, indices = torch.sort(a, descending=True)
            sorted_a_sum = torch.cumsum(sorted_a, dim=1)
            mask = sorted_a_sum>=0.65
            sorted_a[mask] = -1000
        chunk_sorted_a = sorted_a.softmax(dim=1)
        print('Attention_weight:', torch.max(chunk_sorted_a, 1)[0])
        print('Index:',indices[:,0])
        reorder_confounded = dic_z[indices]
        # z_hat = chunk_sorted_a.unsqueeze(2) * reorder_confounded.unsqueeze(0)
        z_hat = reorder_confounded * chunk_sorted_a.unsqueeze(-1)
        return z_hat
    
    def topk_similarity(self, y, dic_z):
        with torch.no_grad():
            norm_y = nn.functional.normalize(y, dim=1)
            norm_z = nn.functional.normalize(dic_z, dim=1)
            similarity = torch.matmul(norm_y, norm_z.t())
            topk = 10
            sorted_a, indices = torch.topk(similarity, topk)
            prior = torch.tensor(np.ones(topk) / (topk),
                                  dtype=torch.float).cuda()
            normoalized_a = nn.functional.softmax(sorted_a, dim=1)
            
            print('Attention_weight:', torch.max(normoalized_a, 1)[0])
            print('Index:',indices[:,0])
            # print(prior)
            reorder_confounded = dic_z[indices]
            z_hat = reorder_confounded * normoalized_a.unsqueeze(-1)
            
            z = torch.matmul(prior.unsqueeze(0), z_hat).squeeze(1)
        return z

    def forward(self, x, bag_idx, inner_idx, old_backbone, imgs):
        device = x.get_device()
        dic_z = self.dic.to(device)
        prior = self.prior.to(device)

       # 1. old backbone
        # if old_backbone:
        #     selected_f = old_backbone(imgs)
        # else:
        #     selected_f = self.select_feature(bag_idx, inner_idx)
        # _, z = self.z_dic(selected_f, dic_z, prior)
        # inter_x = torch.cat((x, z), 1)

        # 2. own feature
        inter_x, z = self.z_dic(x, dic_z, prior)
        causal_logits_list = self.causal_score(inter_x)
        # cluster_preds = self.cluster_score(z)
        # cluster_label = self.select_feature(bag_idx, inner_idx)
        # return causal_logits_list, cluster_preds, cluster_label
        return causal_logits_list


    def z_dic(self, y, dic_z, prior):
        """
        Please note that we computer the intervention in the whole batch rather than for one object in the main paper.
        """
        length = y.size(0)
        if length == 1:
            print('debug')
        # ##1.0 attention
        attention = torch.mm(self.Wy(y), self.Wz(dic_z).t()) / (self.embedding_size ** 0.5)
        # attention = torch.mm(nn.functional.normalize(self.Wy(y), dim=1), nn.functional.normalize(self.Wz(dic_z), dim=1).t()) 
        #[N*M] = [N*D]*[D*M] N:Batch size, D: feature dimension, M:number of confounder
        attention = F.softmax(attention, 1)
        
        print('Attention_weight:', torch.max(attention,1))

        #[N*M]
        ## 1.0 ori_atten
        z_hat = attention.unsqueeze(2) * dic_z.unsqueeze(0)
        ## 1.1 trunk_atten
        # z_hat = self.trunk_attention(attention, dic_z)

        #[N*M*D] =[N*M*1]*[1*M*D]
        z = torch.matmul(prior.unsqueeze(0), z_hat).squeeze(1)
        # [N*1*D]=[1*M]matmul[N*M*D]   .sqz(1) = [N*D]
        # xz = torch.cat((y.unsqueeze(1).repeat(1, length, 1), z.unsqueeze(0).repeat(length, 1, 1)), 2).view(-1, 2*y.size(1))
        
        # z = self.topk_similarity(y, dic_z) 
        xz = torch.cat((y, z), 1)
        # detect if encounter nan
        if torch.isnan(xz).sum():
            print(xz)
        return xz, z

    def select_feature(self, bag_idx, inner_idx):
        with torch.no_grad():
            maskinner = torch.eq(bag_idx.unsqueeze(1), torch.tensor(self.bag_idx).view(1,-1))
            maskbag = torch.eq(inner_idx.unsqueeze(1), torch.tensor(self.inner_idx).view(1,-1))
            mask = maskinner&maskbag
            selected_idx = torch.nonzero(mask)[:,1]
            selected_f = self.feature[selected_idx]
            cluster_idx = self.kmeans.index.search(selected_f, 1)[1]
        # return torch.from_numpy(cluster_idx).squeeze(1)
        return torch.from_numpy(selected_f).cuda()

def l2_normalize(tensor):
    norm = tensor.norm(p=2, dim=1).unsqueeze(-1)
    return tensor.div(norm.expand_as(tensor))


if __name__=='__main__':
    backbone = Dense_small()
    net = ClsNet(backbone, 2).cuda()
    print(net)
    with torch.no_grad():
        x = torch.randn([32,3,224,224]).cuda()
        print(net(x))
        print('ok')