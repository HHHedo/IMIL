import os, sys
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import numpy as np
from utils.utility import GaussianBlurConv
import random
class MemoryBank(object):
    """
    This is a general type of MIL memory bank.
    The memory bank could be a general two-level container (bag-level and instance-level).
    I recommend using structured memory bank (`torch.tensor` or `numpy.array`).
    See `TensorMemoryBank` for more details.

    Notes:
        1. update(): put values into memory bank.
        2. get_rank(): get the relative rank (0~1) 
        3. get_weight(): get the loss weight given relative rank and ratio.
        4. different pooling method (max and average)
        5. plug-and-play cal weight function enabled (updated 2020.2.23)
            currently, weight_func must accept (bag_index, inner_index, ranks, nodule_ratios)
            as input argument.
    
    Args:
        dictionary: (obj) a two-level container (bag level and ins level)
        mmt: (float) momentum for updating the dictionary
        weight_func: (callable) weight calculating function.
    """
    def __init__(self, dictionary={}, mmt=0.75, weight_func=None):
        super(MemoryBank, self).__init__()
        self.dictionary = dictionary
        self.mmt = mmt
        self.weight_func = None

    def state_dict(self):
        return self.dictionary

    def load(self, load_dir=None, resume=None):
        """
        Load memory bank according to resume config and logger.
        Note that logger is used to help loading (logger.load() function should be implemented) (TODO)
        """
        if load_dir is not None:
            if resume > 0:
                self.dictionary = torch.load(os.path.join(load_dir, "res{}.pth".format(resume)))
                print('Mb to be loaded res{}.pth'.format(resume))
                ##BC code, if dictionary, find "memory_bank" key
                if isinstance(self.dictionary, dict):
                    self.dictionary = self.dictionary["memory_bank"]
                    print('Mb loaded res{}.pth'.format(resume))


    def update(self, bag_index, inner_index, instance_preds, epoch=None):
        """
        Since this is a general type memory bank, should check the type of
        memory bank and use different indexing operation (TODO)

        Args:
            bag_index: (list) bag key
            inner_index: (list) instance key
            instance_preds: (torch.Tensor) result of prediction
        """
        # .view(-1) deleted, add cls_num
        result = instance_preds.detach()
        ##if is a dict or list, use for loop
        if isinstance(self.dictionary, dict) or isinstance(self.dictionary, list):
            for k in range(len(inner_index)):
                self.dictionary[bag_index[k]][inner_index[k]][:] = self.mmt * \
                    self.dictionary[bag_index[k]][inner_index[k]][:] + \
                    (1 - self.mmt) * result[k]
        elif isinstance(self.dictionary, torch.Tensor):
            self.dictionary[bag_index, inner_index] = self.mmt * \
                    self.dictionary[bag_index, inner_index] + \
                    (1 - self.mmt) * result

    def _sort(self):
        raise NotImplementedError

    def get_rank(self, bag_index, inner_index):
        """
        get relative rank of instances inside a bag

        relative rank = absolute rank / (length - 1)
        
        Example:
            self.dictionary is [[0.1,0.2,0.3,0.05],
                                [0.3,0.6,0.2,0.1]]
            
            get_rank([0,1], [0,0]) return [0.33, 0.66]
            since 0.1 in first row is the second lowest element (absolute rank=1) 
            and 0.3 is the third lowest element in the second row.
        
        Notes:
            only implemented in sub-class. (2020.2.4)
        """
        raise NotImplementedError

    def get_weight(self, bag_index, inner_index, nodule_ratios, **kwargs):
        """
        This function calls get_rank() and cal_weight() to get the weight.
        """
        ranks = self.get_rank(bag_index, inner_index)
        if self.weight_func is None:
            weights = self.cal_weight(ranks, nodule_ratios)
        else:
            weights = self.weight_func(ranks, nodule_ratios, bag_index, inner_index, **kwargs)
        return weights

    def cal_weight(self, ranks, nodule_ratios):
        """
        This is a plug-in function, could be modified/replaced in the future.
        Args:
            ranks: (torch.Tensor) [M, ] relative ranks of the prediction.
            nodule_ratios: (torch.Tensor) [M, ] nodule ratios to calculate the RCE weight.
            (Reference: Rectified Cross Entropy Loss)

        Notes:
            1. Currently we use threshold linear function
            2. It is promised that if nodule ratio is zero, the weight is approximately zero.
        """
        weights = (ranks / (1 - nodule_ratios.mean())).clamp(max=1.0, min=0.0)
        weights[nodule_ratios<1e-6] = 1.0
        return weights

    def max_pool(self, bag_lengths):
        return_list = []
        for idx, bag in enumerate(self.dictionary):
            # pred = max(bag).cpu()
            pred = torch.max(bag[:bag_lengths[idx]].cpu(), 0)[0]
            return_list.append(pred)
        
        return torch.stack (return_list)

    def avg_pool(self, bag_lengths):
        return_list = []
        for idx, bag in enumerate(self.dictionary):
            # pred = sum(bag[:bag_lengths[idx]].cpu()) / bag_lengths[idx]
            # print(bag[:bag_lengths[idx]].cpu(),sum(bag[:bag_lengths[idx]].cpu()) , sum((bag[:bag_lengths[idx]].cpu()>0.5)))
            pred = torch.mean(bag[:bag_lengths[idx]].cpu(), 0)
            return_list.append(pred)
        
        return torch.stack(return_list)

    def voting_pool(self, bag_lengths):
        return_list = []
        for idx, bag in enumerate(self.dictionary):
            # pred = sum((bag[:bag_lengths[idx]].cpu()>0.5)) / bag_lengths[idx]
            pred = torch.mean((bag[:bag_lengths[idx]].cpu()>.5).float(),0)
            return_list.append(pred)
        return torch.stack(return_list)

    def to_list(self, bag_lengths):
        return_list = []
        for idx, bag in enumerate(self.dictionary):
            return_list.extend(list(bag[:bag_lengths[idx]].cpu()))
        return return_list

class TensorMemoryBank(MemoryBank):
    """
    torch.Tensor version of memory bank. Since it is structured data, something is different:
        1. A global rank Tensor (sized [N, M]) is maintained, where N is bag number
         and M is the max bag size
        2. Tensor based implementation of ranking.
        3. An initialize function for the memory bank embedded.

    Args:
        bag_num: (int) the total number of bag
        max_ins_num: (int) maximum bag length. This could be equal to
            max(bag_lens)
        bag_lens: (list of int) bag length for each bag, sized [bag_num, ]
    """
    def __init__(self, bag_num, max_ins_num=None, bag_lens=None, cls_num=None, mmt=0.75, weight_func=None, max_epoch=200, cur_epoch=0, device=torch.device("cuda")):
        self.device = device
        self.bag_num = bag_num
        self.bag_lens = torch.tensor(bag_lens).view(bag_num, 1)
        self.max_ins_num = max_ins_num
        if self.max_ins_num is None and self.bag_lens is None:
            raise Exception
        elif self.bag_lens is None:
            self.bag_lens = torch.tensor([max_ins_num] * bag_num).view(bag_num, 1)
        elif self.max_ins_num is None:
            self.max_ins_num = max(self.bag_lens)

        init_bank = self.init_bank(bag_num, max_ins_num, cls_num).to(self.device)
        super(TensorMemoryBank, self).__init__(init_bank, mmt, weight_func)
        self.max_epoch = max_epoch
        self.cur_epoch = cur_epoch
        self.init_rank()

    def update_epoch(self):
        if self.cur_epoch < self.max_epoch:
            self.cur_epoch+=1

    def init_bank(self, bag_num, max_ins_num, cls_num):
        """
        Since each bag(a row) has a valid length, elements
        that are out of index are marked as a pre-defined value (-1.0 since no score < 0)
        """
        # if cls_num is None:
        #     init_tensor = torch.ones([bag_num, max_ins_num]).float()
        #     for idx, bag_len in enumerate(self.bag_lens):
        #         init_tensor[idx, bag_len:] = -1.0
        print(bag_num, max_ins_num, cls_num)
        init_tensor = torch.ones([bag_num, max_ins_num, cls_num]).float()
        for idx, bag_len in enumerate(self.bag_lens):
            init_tensor[idx, bag_len:,:] = -1.0

        return init_tensor

    # def init_coor(self, bag_num, max_ins_num):
    #     """
    #     The same as init_bank
    #     """
    #     coor_x_tensor = torch.zeros([bag_num, max_ins_num]).float()
    #     coor_y_tensor = torch.zeros([bag_num, max_ins_num]).float()
    #     for idx, bag_len in enumerate(self.bag_lens):
    #         coor_x_tensor[idx, bag_len:] = -1.0
    #         coor_y_tensor[idx, bag_len:] = -1.0
    #
    #     return coor_x_tensor, coor_y_tensor

    def init_rank(self):
        self.rank_tensor = torch.ones_like(self.dictionary).cuda()

    def update_rank(self):
        """
        Thanks qsf for the idea of implementing this!
        This function update the relative rank for memory bank (sized [M,N]).
        Notice that for each bag the bag length is not the same. The last few
        elements in each row would be negative.

        """
        for i in range(self.dictionary.shape[2]):
            abs_ranks = self.dictionary[:,:,i].argsort(dim=1).argsort(dim=1)
            # argsort: achieve the arg by the value from min to max
            ##ignore last few elements by setting them as minors
            abs_ranks = abs_ranks.cuda() - self.max_ins_num + self.bag_lens.cuda()
            relative_ranks = abs_ranks.float() / (self.bag_lens-1).float().cuda()
            self.rank_tensor[:,:,i] = relative_ranks

    def get_rank(self, bag_index, inner_index):
        return self.rank_tensor[bag_index, inner_index]


class RCETensorMemoryBank(TensorMemoryBank):
    """
    Our implementation of RCE Loss Memory Bank;
    """
    def __init__(self, bag_num, max_ins_num=None, bag_lens=None, cls_num=None, mmt=0.75, mean_ratios=0.5, min_ratios=0.5):
        super(RCETensorMemoryBank, self).__init__(bag_num, max_ins_num, bag_lens, cls_num, mmt)
        self.mean_ratios = mean_ratios
        self.min_ratios = min_ratios
        # if cls_num == 1:
        #     self.dictionary = self.dictionary.squeeze(-1)
        #     print(self.dictionary.shape)
        # else:
        #     raise NotImplementedError


    def get_weight(self, bag_index, inner_index, nodule_ratios, preds=None, cur_epoch=None):
        """
        Notice that we just need mean-ratios and min-ratios. The parameter of this function
        might be different from MRCE and SPCE.
        """
        ranks = self.get_rank(bag_index, inner_index)
        weights = self.cal_weight(bag_index, inner_index, ranks, nodule_ratios, preds)
        return weights, None

    def cal_weight(self, bag_index, inner_index, ranks, nodule_ratios, preds):
        """
        The RCE calculating function.
        Reference: Rectified Cross-Entropy and Upper Transition Loss for 
                    Weakly Supervised Whole Slide Image Classifier.
        """
        weights = torch.ones_like(ranks).float()
        lower_bound = 1 - self.mean_ratios
        upper_bound = 1 - self.min_ratios

        ## RCE function
        weights = torch.clamp((ranks - lower_bound) / (upper_bound - lower_bound), min=0.0, max=1.0)
        ## Re-weight negative instances
        weights[nodule_ratios<1e-6] = 1.0
        return weights


class MRCETensorMemoryBank(TensorMemoryBank):
    """
    re-implement the cal_weight function (provided by ltc).
    """
    def __init__(self, bag_num, max_ins_num=None, bag_lens=None, mmt=0.75):
        super(MRCETensorMemoryBank, self).__init__(bag_num, max_ins_num, bag_lens, mmt)

    def get_weight(self, bag_index, inner_index, nodule_ratios, preds=None):
        ranks = self.get_rank(bag_index, inner_index)
        weights = self.cal_weight(bag_index, inner_index, ranks, nodule_ratios, preds)
        return weights

    def cal_weight(self, bag_index, inner_index, ranks, nodule_ratios, preds):
        """
        The weight calculating protocol is: Depict those with score lower than 0.5
        """
        weights = torch.ones_like(ranks).float()
        mask1 = (self.dictionary[bag_index, inner_index] < 0.5)
        weights[mask1] = 0.0
        weights[nodule_ratios<1e-6] = 1.0
        return weights


class MRCEV3TensorMemoryBank(TensorMemoryBank):
    """
    re-implement the cal_weight function (provided by ltc).
    """
    def __init__(self, bag_num, max_ins_num=None, bag_lens=None, mmt=0.75):
        super(MRCEV3TensorMemoryBank, self).__init__(bag_num, max_ins_num, bag_lens, mmt)

    def get_weight(self, bag_index, inner_index, nodule_ratios, preds=None):
        ranks = self.get_rank(bag_index, inner_index)
        weights = self.cal_weight(bag_index, inner_index, ranks, nodule_ratios, preds)
        return weights

    def cal_weight(self, bag_index, inner_index, ranks, nodule_ratios, preds):
        """
        Our weight calculating function (v3)

        """
        weights = torch.ones_like(ranks).float()
        th1 = 1- nodule_ratios
        th2 = ((self.dictionary<0.5) & (self.dictionary>=0)).float().mean(dim=1)
        weights[(ranks>=th2) & (ranks<th1)] = (ranks - th2) / (th1 - th2)
        weights[ranks<th2] = 0.0
        weights[nodule_ratios<1e-6] = 1.0
        return weights


class SPCETensorMemoryBank(TensorMemoryBank):
    """
    Our self-paced CE Tensor Memory Bank;
    Notes:
        1. cal_weight function is epoch-aware; linear protocol
        2. The relative rank is re-implemented (rank 0 start from those with score > 0.5)
    """
    def __init__(self, bag_num, max_ins_num=None, bag_lens=None, cls_num=None, mmt=0.75, max_epoch=200, cur_epoch=0):
        super(SPCETensorMemoryBank, self).__init__(bag_num, max_ins_num, bag_lens, cls_num, mmt, None, max_epoch, cur_epoch)

    def get_weight(self, bag_index, inner_index, nodule_ratios, preds=None):
        ranks = self.get_rank(bag_index, inner_index)
        weights = self.cal_weight(bag_index, inner_index, ranks, nodule_ratios, preds)
        return weights

    def cal_weight(self, bag_index, inner_index, ranks, nodule_ratios, preds):
        weights = torch.ones_like(ranks).float()
        beta = self.cur_epoch / self.max_epoch    
        weights = (ranks / (beta+1e-6)).clamp(min=0.0, max=1.0)
        weights[nodule_ratios<1e-6] = 1.0
        return weights

    def update_rank(self):
        """
        Thanks qsf for the idea of implementing this!
        This function update the relative rank for memory bank (sized [M,N]).
        Notice that for each bag the bag length is not the same. The last few
        elements in each row would be negative.

        """
        # abs_ranks = self.dictionary.argsort(dim=1).argsort(dim=1)
        # ##ignore last few elements by setting them as minors
        # abs_ranks = abs_ranks - self.max_ins_num
        # abs_ranks = (abs_ranks.T + (self.dictionary>0.5).sum(1))
        # relative_ranks = abs_ranks.float() / ((self.dictionary>0.5).sum(1).float() - 1).clamp(min=1.0)
        # self.rank_tensor = relative_ranks.T
        return 0


class CaliTensorMemoryBank(TensorMemoryBank):
    """
    Our implementation of RCE Loss Memory Bank;
    """
    def __init__(self, bag_num, max_ins_num=None, bag_lens=None, cls_num=None, mmt=0.75, ignore_ratio=0, total_epoch=10):
        super(CaliTensorMemoryBank, self).__init__(bag_num, max_ins_num, bag_lens, cls_num, mmt)
        self.ignore_ratio = ignore_ratio
        self.total_epoch = total_epoch


    def get_mean(self, bag_index, inner_index):
        mean_list = []
        for k in range(len(inner_index)):
            bag_preds = self.dictionary[bag_index[k]]
            # some -1 should be ignored
            valid_preds = bag_preds[bag_preds > 0]
            mean_list.append(valid_preds.mean())
        mean_tensor = torch.stack(mean_list)
        return mean_tensor.unsqueeze(-1)

    def get_weight(self, bag_index, inner_index, nodule_ratios, preds, cur_epoch):
        # cal the mean preds of bag for each instance
        mean_tensor = self.get_mean(bag_index, inner_index)
        # setting the negative mean preds to be 1e-6
        # mean_tensor[nodule_ratios < 1e-6] = 1e-6
        mean_tensor[nodule_ratios < 1e-6] = -1
        # calibration
        with torch.no_grad():
            # making it be negative to select the topK smallest
            tmp_preds = preds.detach() - mean_tensor
            # k is updated with epoch #TODO whether preds[nodule_ratios >= 1e-6]
            ignore_num = int(self.ignore_ratio*(nodule_ratios > 1e-6).sum())
            k = min(int((cur_epoch/self.total_epoch)*ignore_num), ignore_num)
            _, idx = torch.topk(tmp_preds, k, dim=0, largest=False)
            weight = torch.ones_like(preds)
            if idx.shape[1] != 0:
                for i in idx:
                    weight[i] = 0
            pos_weight = 1/(1 - min((cur_epoch/self.total_epoch)*self.ignore_ratio, self.ignore_ratio))
        return weight, torch.tensor([pos_weight]).cuda()


class PBTensorMemoryBank(TensorMemoryBank):
    """
    Our implementation of RCE Loss Memory Bank;
    """

    def __init__(self, bag_num, max_ins_num=None, bag_lens=None, cls_num=None, mmt=0.75,
                 bag_idx_list=[], inner_index_list=[], c_x=None, c_y=None, bag_pos_ratio=None, update_iter=2,
                 ):
        super(PBTensorMemoryBank, self).__init__(bag_num, max_ins_num, bag_lens, cls_num, mmt)
        self.bag_idx_tensor = torch.tensor(bag_idx_list)
        self.inner_index_tensor = torch.tensor(inner_index_list)
        if c_x is not None:
            self.c_x_tensor = torch.tensor(c_x)
            self.H = self.c_x_tensor.max() + 1

        if c_y is not None:
            self.c_y_tensor = torch.tensor(c_y)
            self.W = self.c_y_tensor.max() + 1
        self.bag_pos_ratio_tensor = torch.stack(bag_pos_ratio)

        self.update_iter = update_iter
        # tmp_dict for not directly updating
        self.tmp_dict = self.dictionary.clone()
        self.N = self.bag_idx_tensor.max() + 1

        self.gaussian_conv = GaussianBlurConv(channels=self.N).cuda()
        self.ignore_num = torch.zeros(cls_num)
        self.ignore_goon = torch.tensor([True] * cls_num)

    def Gaussian_smooth(self):
        '''
        mmt = 0 ,
        update_iter = 2:
            update + using Gaussian smooth

        '''
        with torch.no_grad():
            for i in range(self.tmp_dict.shape[2]):
                matrix_preds = torch.zeros([self.N, self.H, self.W]).cuda()
                # preds = self.tmp_dict
                matrix_preds[self.bag_idx_tensor, self.c_x_tensor, self.c_y_tensor] = \
                    self.tmp_dict[:,:,i][self.bag_idx_tensor, self.inner_index_tensor]
                smoothed_preds = self.gaussian_conv(matrix_preds)
                self.dictionary[:,:,i][self.bag_idx_tensor, self.inner_index_tensor] = \
                    smoothed_preds[self.bag_idx_tensor, self.c_x_tensor, self.c_y_tensor]

    def update_tmp(self, bag_index, inner_index, instance_preds, epoch=None):
        """
        Using self.tmp_dict for tmp updating
        """
        # if (epoch+1) % self.update_iter == 0:

        result = instance_preds.detach() # no .view(-1)
        ##if is a dict or list, use for loop
        if isinstance(self.dictionary, dict) or isinstance(self.dictionary, list):
            for k in range(len(inner_index)):
                self.tmp_dict[bag_index[k]][inner_index[k]] = self.mmt * \
                    self.tmp_dict[bag_index[k]][inner_index[k]] + \
                    (1 - self.mmt) * result[k]
        elif isinstance(self.dictionary, torch.Tensor):
            self.tmp_dict[bag_index, inner_index] = self.mmt * \
                    self.tmp_dict[bag_index, inner_index] + \
                    (1 - self.mmt) * result
        else:
            print('Not updating this epoch!')

    # def get_weight(self, bag_index, inner_index, nodule_ratios, preds, cur_epoch):
    #     ranks = self.get_rank(bag_index, inner_index)
    #     weights = self.cal_weight(bag_index, ranks, nodule_ratios, cur_epoch)
    #     # print(weights[weights != 1])
    #     return weights.unsqueeze(-1), None
    #
    # def cal_weight(self, bag_index, ranks, nodule_ratios, cur_epoch):
    #     """
    #     The RCE calculating function.
    #     Reference: Rectified Cross-Entropy and Upper Transition Loss for
    #                 Weakly Supervised Whole Slide Image Classifier.
    #     """
    #     weights = torch.ones_like(ranks).float()
    #     bound = 1 - self.bag_pos_ratio_tensor[bag_index]
    #     # print(ranks, cur_epoch)
    #     weights[ranks < bound.cuda()] = 0
    #     # ## Re-weight negative instances
    #     weights[nodule_ratios < 1e-6] = 1.0
    #     return weights

    def select_samples(self):
        #TODO rank tensor for multi class
        ratio_tensor = 1 - self.bag_pos_ratio_tensor.unsqueeze(-1).repeat(1, self.max_ins_num).cuda()
        # pos_samples = (self.rank_tensor >= ratio_tensor)[self.bag_pos_ratio_tensor>0]
        # neg_samples = self.dictionary[self.bag_pos_ratio_tensor==0] >0
        pos_samples = (self.rank_tensor.squeeze(-1) >= ratio_tensor)[self.bag_pos_ratio_tensor > 0]
        neg_samples = (self.dictionary[self.bag_pos_ratio_tensor==0] >0).squeeze(-1)
        selected_idx = torch.cat((pos_samples, neg_samples)).float()
        return selected_idx

    def get_mean(self): # mean preds for each bag
        mean_list = []
        for k in range(self.bag_num):
            bag_preds = self.dictionary[k]
            # some -1 should be ignored
            valid_preds = bag_preds[bag_preds > 0]
            mean_list.append(valid_preds.mean())
        mean_tensor = torch.stack(mean_list)
        return mean_tensor.unsqueeze(-1)

    def select_topk(self, bag_labels):
        # _, idx = torch.topk(self.dictionary, k=50, dim=1)
        # selected_idx = torch.zeros_like(self.dictionary).cuda()
        # # selected_idx[idx] = 1
        # selected_idx.scatter_(1, idx, 1)
        # selected_idx[self.dictionary == -1] = 0
        selectd_idx_tensor = torch.zeros_like(self.dictionary)
        for i in range(self.dictionary.shape[2]):
            _, idx = torch.topk(self.dictionary[:,:,i], k=25, dim=1)
            selected_idx = torch.zeros_like(self.dictionary[:,:,i]).cuda()
            # selected_idx[idx] = 1
            selected_idx.scatter_(1, idx, 1)
            if (self.dictionary == -1).sum() != 0 :
                selected_idx[self.dictionary == -1] = 0
            selected_idx[bag_labels[:, i] == 0] = 0
            selectd_idx_tensor[:,:,i] = selected_idx
        selected_idx = (torch.sum(selectd_idx_tensor,2)>0).int()
        return selected_idx

    def get_mean_(self, dict): # mean preds for each bag
        mean_list = []
        for k in range(self.bag_num):
            bag_preds = dict[k]
            # some -1 should be ignored
            valid_preds = bag_preds[bag_preds > 0]
            mean_list.append(valid_preds.mean())
        mean_tensor = torch.stack(mean_list)
        mean_tensor = 0.2*mean_tensor + 0.8*mean_tensor.mean()
        return mean_tensor.unsqueeze(-1)


    def select_calibrated(self, K, thres, step, bag_labels):
        #  select top k-i for 20 classes:
        if K>0:
            self.ignore_num[self.ignore_goon] = K
        print(self.ignore_num)
        selectd_idx_tensor = torch.zeros_like(self.dictionary)
        selectd_idx_label_tensor = torch.zeros_like(self.dictionary)
        selectd_weight_tensor = torch.zeros_like(self.dictionary)
        new_selected_preds_tensor = torch.zeros(self.dictionary.shape[2]) # mean of mean of each class
        new_selected_preds_list = [] # mean of selected of all classes
        for i in range(self.dictionary.shape[2]): # for each class
            # calculate the k
            pos_ins_num = (self.bag_lens[self.bag_pos_ratio_tensor > 0].squeeze(-1)*bag_labels[self.bag_pos_ratio_tensor > 0][:,i]).sum()
            k = int(self.ignore_num[i] * pos_ins_num)
            dict = self.dictionary[:, :, i]
            selected_idx = torch.ones_like(dict).cuda()
            selected_idx_label = torch.ones_like(dict).cuda() # all 1
            selected_idx_label[bag_labels[:, i] == 0] = 0 # not in class = 0
            selected_weight = torch.ones_like(dict).cuda()
            # k = int(0.05 * pos_ins_num)
            # print('{:.3}%/{} Ignored samples.'.format(k / pos_ins_num.float() * 100, k))
            if k != 0:
                # Choose the top-k for positive and all negative instances

                mean_preds = self.get_mean_(dict)
                calibrated_preds = dict / mean_preds
                pos_calibrated_preds = calibrated_preds[bag_labels[:,i] > 0] # postive bag with label 1 in this class
                pos_calibrated_preds_valid = pos_calibrated_preds[pos_calibrated_preds > 0] # valid preds in postive bag
                k_preds, _ = torch.topk(pos_calibrated_preds_valid, k, dim=0, largest=False) # find the value of the k-th instance
                # selecting instance
                pos_selected = calibrated_preds > k_preds[-1] # select those higher than T of this class
                selected_idx = pos_selected.float()
                selected_idx[bag_labels[:,i] == 0] = 1 #  (not ignore but) keep those dont belong to this class
                #label change
                # selected_idx_label = pos_selected.float()  # larger is 1
                # selected_idx_label[bag_labels[:,i] == 0] =0 #  those dont belong to this class is still 0 even larger than T
                # assigning weight
                neg_selected = calibrated_preds < k_preds[-1]
                selected_weight[neg_selected] = 0.5
                selected_weight[bag_labels[:,i] == 0] = 1
                # neg_selected = calibrated_preds[self.bag_pos_ratio_tensor == 0] > 0
                # selected_idx = torch.cat((pos_selected, neg_selected)).float()

                new_selected_preds = pos_calibrated_preds_valid[(pos_calibrated_preds_valid<= k_preds[-1])&
                                                                (pos_calibrated_preds_valid >= k_preds[-int(step * pos_ins_num)])].mean()
                new_selected_preds_list.append(pos_calibrated_preds_valid[(pos_calibrated_preds_valid<= k_preds[-1])&
                                                                (pos_calibrated_preds_valid >= k_preds[-int(step * pos_ins_num)])])

                new_selected_preds_tensor[i] = new_selected_preds
            selectd_idx_tensor[:, :, i] = selected_idx
            selectd_idx_label_tensor[:, :, i] = selected_idx_label
            selectd_weight_tensor[:,:,i] = selected_weight
        selectd_idx_tensor[self.dictionary == -1] = 0
        selected_idx = (torch.sum(selectd_idx_tensor, 2) > 0).int()
        pos_capreds_new = new_selected_preds_tensor
        print("Newly choosen preds", pos_capreds_new)
        print('labels num',selectd_idx_label_tensor.sum())
        print('weight sum', selectd_weight_tensor.sum())
        self.ignore_goon = (pos_capreds_new < thres)*self.ignore_goon
        print( self.ignore_goon)
        # self.ignore_num = k
        # if len(new_selected_preds_list) != 0:
        #     pos_capreds_new = torch.cat(new_selected_preds_list).mean()
        #     print("Newly choosen preds", pos_capreds_new)
        # if configs.ignore_goon:
        #     # new_selected_idx = ((calibrated_preds <= k_preds[-1]) & \
        #     #                     (calibrated_preds > k_preds[-int(configs.ignore_step * pos_ins_num)])).float()
        #     # pos_selected_idx = new_selected_idx[self.memory_bank.bag_pos_ratio_tensor > 0]
        #     # pos_capreds_new = (pos_calibrated_preds[pos_selected_idx == 1]).mean()
        #     # self.logger.log_string('Mean calibrated preds is {}, from ({}  to {}] with {:.3} and {:.3}.'
        #     #                        .format(pos_capreds_new, -int(configs.ignore_step * pos_ins_num), -1,
        #     #                                k_preds[-int(configs.ignore_step * pos_ins_num)], k_preds[-1]))
        #
        #     # top_selected_idx = ((calibrated_preds > k_preds[0]) & \
        #     #                     (calibrated_preds <= k_preds[int(configs.ignore_step * pos_ins_num) - 1])).float()
        #     # top_selected_idx = top_selected_idx[self.memory_bank.bag_pos_ratio_tensor > 0]
        #     # top_capreds_new = (pos_calibrated_preds[top_selected_idx == 1]).mean()
        #     # print('Top Mean calibrated preds is {}, from ({}  to {}] with {:.3} and {:.3}.'
        #     #       .format(top_capreds_new, 0, int(configs.ignore_step * pos_ins_num) - 1,
        #     #               k_preds[0], k_preds[int(configs.ignore_step * pos_ins_num) - 1]))
        #     if pos_capreds_new > ignore_thres and (epoch - epoch_thres) == 1:
        #         epoch_thres += 1
        #     elif pos_capreds_new < ignore_thres:
        #         self.memory_bank.ignore_num = k
        #     else:
        #         configs.ignore_goon = False
        return pos_capreds_new, selected_idx, selectd_idx_label_tensor, selectd_weight_tensor



