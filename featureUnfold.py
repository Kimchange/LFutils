import torch
import torch.nn.functional as F

Tensor = torch.Tensor
def get_index_shift(l,s):
    index = list(range(l))
    x = index[s-1::-1]
    y = index[:s-1:-1]
    return torch.tensor( list( reversed(x+y) ))
def featureUnfoldlf(input, kernel_size, dilation=1, padding=0, stride=1):
    ret = F.pad(input, 2*len(input.shape[2:])*[kernel_size//2], )
    # ret = ret.expand(input.shape[0], kernel_size**len(ret.shape[2:]) * ret.shape[1], *ret.shape[2:]).contiguous()
    ret = ret.repeat(tuple([1, kernel_size**len(input.shape[2:])] + len(input.shape[2:]) *[1]))
    ret = ret.view(input.shape[0], kernel_size**len(input.shape[2:]), input.shape[1], *ret.shape[2:])
    # print(ret.shape)
    ret = ret.transpose(1,2)
    # print(ret.shape)
    ret = ret.contiguous()
    ret = ret.view(input.shape[0], kernel_size**len(input.shape[2:]) * input.shape[1], *ret.shape[3:])
    # print(ret.shape)
    C = 0
    for u in range(kernel_size):
        idx_u = get_index_shift(ret.shape[2], u-kernel_size//2)
        for v in range(kernel_size):
            idx_v = get_index_shift(ret.shape[3], v-kernel_size//2)
            for x in range(kernel_size):
                idx_x = get_index_shift(ret.shape[2], x-kernel_size//2)
                for y in range(kernel_size):
                    idx_y = get_index_shift(ret.shape[2], y-kernel_size//2)
                    u_, v_, x_, y_ = torch.meshgrid(idx_u, idx_v, idx_x, idx_y)
                    # ret[:,C:C+input.shape[1],:,:] = ret[:,C:C+input.shape[1],u_,v_]
                    # ret[:,C,:,:] = ret[:,C,u_,v_]
                    # ret[:,C+input.shape[1],:,:] = ret[:,C+input.shape[1],u_,v_]
                    ret[:,C::kernel_size**len(input.shape[2:]),:,...] = ret[:,C::kernel_size**len(input.shape[2:]),u_,v_,x_,y_]
                    
                    # print(ret[:,C,:,:])
                    C += 1
    print(ret.shape)
    ret = ret[:,:,kernel_size//2:kernel_size//2+input.shape[2], kernel_size//2:kernel_size//2+input.shape[3], \
        kernel_size//2:kernel_size//2+input.shape[2], kernel_size//2:kernel_size//2+input.shape[3]]
# & C:/Users/lab532/AppData/Local/Programs/Python/Python39/python.exe -i e:/code/utils/featureUnfold.py
    
    return ret

def featureUnfold(input, kernel_size, dilation=1, padding=0, stride=1):
    input_pad = F.pad(input, 2*len(input.shape[2:])*[kernel_size//2], )
    ret = torch.cat([input_pad[:,c:c+1,u:u+input.shape[2], v:v+input.shape[3],x:x+input.shape[4],y:y+input.shape[5],] \
                     for c in range(input.shape[1]) \
                     for u in range(kernel_size) \
                     for v in range(kernel_size) \
                     for x in range(kernel_size) \
                     for y in range(kernel_size) ],1)
    return ret

def featureUnfoldnd(input, kernel_size, dilation=1, padding=0, stride=1):
    
    input_pad = F.pad(input, 2*len(input.shape[2:])*[kernel_size//2], )
    ret_ = '[input_pad[:,c:c+1,' + ''.join([f'dim{i}:dim{i}+input.shape[{i+2}],' for i in range(len(input.shape[2:]))])+'] '+ \
                    f'for c in range(input.shape[1]) '+ \
                    ''.join([f'for dim{i} in range({kernel_size}) ' for i in range(len(input.shape[2:]))])+']'
    print(ret_)
    # print(input_pad.shape)
    # print(eval('[input_pad[:].shape]'))
    ret = torch.cat(eval(ret_, {"input_pad":input_pad, "input": input}),1)
    print(eval('[input_pad.shape for i in range(2)]',{"input_pad":input_pad}))
##    ret = torch.cat([eval('input_pad[:,c:c+1,' + ''.join([f'dim{i}:dim{i}+input.shape[{i+2}],' for i in range(len(input.shape[2:]))])+']') \
##                     eval(f'for c in range(input.shape[1]) ') \
##                     eval(''.join([f'for dim{i} in range({kernel_size}) ' for i in range(len(input.shape[2:]))   ]) )+ ], 1)
    return ret




input = torch.randn(5,5,5,5,5,5)
myres = featureUnfoldlf(input,3)
myres2 = featureUnfold(input,3)
myres3 = featureUnfoldnd(input,3)
# res = F.unfold(input,3,padding=1).view(
#                 input.shape[0], input.shape[1] * 9, input.shape[2], input.shape[3])
# print(myres==myres2)
torch.sum(abs(myres-myres3))
# grid = Tensor([0,0]).view(1,1,1,2)
# print(grid)
# myres = 0
# res = F.grid_sample(input,grid, mode='nearest', align_corners=False)

    
    
