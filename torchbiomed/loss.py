import torch
from torch.autograd import Function
import torch.nn as nn
from itertools import repeat
import numpy as np

# Intersection = dot(A, B)
# Union = dot(A, A) + dot(B, B)
# The Dice loss function is defined as
# 1/2 * intersection / union
#
# The derivative is 2[(union * target - 2 * intersect * input) / union^2]


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss,self).__init__()


    def forward(self, input, target, save=True):
        # if save:
        #     self.save_for_backward(input, target)
        # eps = 0.000001
        # _, result_ = input.max(4)
        # result_ = torch.squeeze(result_)
        # if input.is_cuda:
        #     result = torch.cuda.FloatTensor(result_.size())
        #     self.target_ = torch.cuda.FloatTensor(target.size())
        # else:
        #     result = torch.FloatTensor(result_.size())
        #     self.target_ = torch.FloatTensor(target.size())
        # result.copy_(result_)
        # self.target_.copy_(target)
        # target = self.target_
        # # intersect = torch.dot(result, target)
        # intersect = torch.sum(torch.mul(result,target))
        # # binary values so sum the same as sum of squares
        # result_sum = torch.sum(result)
        # target_sum = torch.sum(target)
        # union = result_sum + target_sum + (2*eps)
        #
        # # the target volume can be empty - so we still want to
        # # end up with a score of 1 if the result is 0/0
        # IoU = intersect / union
        # print('union: {:.3f}\t intersect: {:.6f}\t target_sum: {:.0f} IoU: result_sum: {:.0f} Dice {:.7f}'.format(
        #     union, intersect, target_sum, result_sum, 2*IoU))
        # # out = torch.FloatTensor(1).fill_(2*IoU)
        # self.intersect, self.union = intersect, union
        # return out

        N = target.size(0)
        smooth = 1

        input_ = input.max(4)[1].float()

        input_flat = input_.view(N, -1)
        target_flat = target.view(N, -1)

        intersection = input_flat * target_flat

        intersection_sum = torch.sum(intersection).float()
        input_sum = torch.sum(input_flat).float()
        target_sum = torch.sum(target_flat).float()

        # loss = 2*torch.div(intersection_sum,(input_sum + target_sum))
        loss = 2 * (intersection_sum) / (input_sum + target_sum)
        loss = 1 - loss / N

        return loss





# class DiceLoss(Function):
#     def __init__(self, *args, **kwargs):
#         pass
#
#     def forward(self, input, target, save=True):
#         if save:
#             self.save_for_backward(input, target)
#         eps = 0.000001
#         _, result_ = input.max(4)
#         result_ = torch.squeeze(result_)
#         if input.is_cuda:
#             result = torch.cuda.FloatTensor(result_.size())
#             self.target_ = torch.cuda.FloatTensor(target.size())
#         else:
#             result = torch.FloatTensor(result_.size())
#             self.target_ = torch.FloatTensor(target.size())
#         result.copy_(result_)
#         self.target_.copy_(target)
#         target = self.target_
#         # intersect = torch.dot(result, target)
#         intersect = torch.sum(torch.mul(result,target))
#         # binary values so sum the same as sum of squares
#         result_sum = torch.sum(result)
#         target_sum = torch.sum(target)
#         union = result_sum + target_sum + (2*eps)
#
#         # the target volume can be empty - so we still want to
#         # end up with a score of 1 if the result is 0/0
#         IoU = intersect / union
#         print('union: {:.3f}\t intersect: {:.6f}\t target_sum: {:.0f} IoU: result_sum: {:.0f} Dice {:.7f}'.format(
#             union, intersect, target_sum, result_sum, 2*IoU))
#         out = torch.FloatTensor(1).fill_(2*IoU)
#         self.intersect, self.union = intersect, union
#         return out
#
#     def backward(self, grad_output):
#         input, _ = self.saved_tensors
#         intersect, union = self.intersect, self.union
#         target = self.target_
#         gt = torch.div(target, union)
#         IoU2 = intersect/(union*union)
#         pred = torch.mul(input[:, 1], IoU2)
#         dDice = torch.add(torch.mul(gt, 2), torch.mul(pred, -4))
#         grad_input = torch.cat((torch.mul(dDice, -grad_output[0]),
#                                 torch.mul(dDice, grad_output[0])), 0)
#         return grad_input , None

def dice_loss():
    return DiceLoss()

# def dice_loss(input, target):
#     return DiceLoss(input, target)

def dice_error(input, target):
    eps = 0.000001
    _, result_ = input.max(1)
    result_ = torch.squeeze(result_)
    if input.is_cuda:
        result = torch.cuda.FloatTensor(result_.size())
        target_ = torch.cuda.FloatTensor(target.size())
    else:
        result = torch.FloatTensor(result_.size())
        target_ = torch.FloatTensor(target.size())
    result.copy_(result_.data)
    target_.copy_(target.data)
    target = target_
    intersect = torch.dot(result, target)

    result_sum = torch.sum(result)
    target_sum = torch.sum(target)
    union = result_sum + target_sum + 2*eps
    intersect = np.max([eps, intersect])
    # the target volume can be empty - so we still want to
    # end up with a score of 1 if the result is 0/0
    IoU = intersect / union
#    print('union: {:.3f}\t intersect: {:.6f}\t target_sum: {:.0f} IoU: result_sum: {:.0f} IoU {:.7f}'.format(
#        union, intersect, target_sum, result_sum, 2*IoU))
    return 2*IoU
