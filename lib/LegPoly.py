import torch
import math

class LegendrePolynomial0(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.new_ones(input.size())

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return input.new_zeros(input.size())
    
class LegendrePolynomial1(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output * input.new_ones(input.size())
    

class LegendrePolynomial2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return 0.5 * (3 * input ** 2 - 1)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output * 3 * input
    
class LegendrePolynomial3(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return 0.5 * (5 * input ** 3 - 3 * input)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output * 0.5 * (15 * input ** 2 - 3)
    

class LegendrePolynomial4(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return 0.125 * (35 * input ** 4 - 30 * input ** 2 + 3)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output * 0.125 * (140 * input ** 3 - 60 * input)
    

class LegendrePolynomial5(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return 0.125 * (63 * input ** 5 - 70 * input ** 3 + 15 * input)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output * 0.125 * (315 * input ** 4 - 210 * input ** 2 + 15)
    
class LegendrePolynomial6(torch.autograd.Function): 
    @staticmethod
    def forward(ctx, input): 
        ctx.save_for_backward(input)
        return 0.0625 * (231 * input**6 - 315 * input**4 + 105 * input**2 - 5)
    
    @staticmethod
    def backward(ctx, grad_output): 
        input, = ctx.saved_tensors
        return grad_output * 0.0625 * (1386 * input**5 - 1260 * input**3 + 210 * input)

