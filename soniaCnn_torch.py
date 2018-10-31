import torch
import torchvision
import torchvision.transforms as transforms


# SOM LAYER
class Som(torch.autograd.Function):
    def __init__(self):
        super(Som, self).__init__()

    @staticmethod
    def forward(ctx, input, weight, bias=None):
        ctx.save_for_backward(input, weight, bias)
        
    
    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
