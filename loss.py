import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self, smooth=0.0001, p=2, reduction='mean'):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction
    def forward(self, predict, target):
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        inter = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        union = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - inter / union

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))

if __name__ == '__main__':
    # to change
    predict = torch.tensor([[1., 2.],
                          [2., 3.]])
    target = torch.tensor([[1., 2.],
                           [2., 3.]])
    criterion = DiceLoss()
    loss = criterion(predict, target)
    print('loss is:', loss)