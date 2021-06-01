import torch
import torch.nn.functional as F

Tensor = torch.Tensor
def unravel_index(index, shape):
    out = []
    for dim in reversed(shape):
        out.append(index % dim)
        index = index // dim
    return tuple(reversed(out))

def ravel_multi_index(multi_index, shape):
    out = 0
    multi_index = list(multi_index)
    for dim in shape[1:]:
        out += multi_index[0]
        out *= dim
        multi_index.pop(0)
    out += multi_index[0]
    return out

def nearestGridSample(
    input: Tensor,
    grid: Tensor,
    mode: str = "nearest",
    padding_mode: str = "zeros",
    align_corners: bool = False,
) -> Tensor:
    # input (batchsize, channel, height(dim0), width(dim1), depth(dim2), dim3, ...)
    # output (batchsize, channel, *outputshape)
    # grid (batchsize, *outputshape, length(dim))
    #
    # output = Tensor(input.shape[:2] + grid.shape[1:-1])
    index = torch.ceil( (grid+1)/2*Tensor(list(input.shape[-1:1:-1]))) - 1
    index = index.flip(-1)
    grid = ravel_multi_index(index.permute((len(index.shape)-1,)+tuple([i for i in range(len(index.shape)-1)]) ), input.shape[2:])
##    print(index)
##    print(grid)
    input = input.view(input.shape[0],input.shape[1],-1)
    return torch.cat([input[b:b+1, :, grid[b,:,...].long()] for b in range(input.shape[0])],0)

# nearestGridSample(torch.ones(2,2,2,2),torch.ones(2,2,2,2))
# index = Tensor([[[0,0],[1,1],[2,2],[3,3],[4,4],],[[0,0],[1,1],[2,2],[3,3],[4,4],]]).long()
# index = Tensor([[[0,0],[1,1],[2,2],[3,3],[4,4],],[[0,0],[1,1],[2,2],[3,3],[4,4],],[[0,0],[1,1],[2,2],[3,3],[4,4],]]).long()
# l = Tensor.arange(25).view(5,5)
# l[index[...,0],index[...,1],]

input = torch.randn(5,3,10,5,4)
grid = torch.rand(5,2,3,2,3)*2-1
# grid = Tensor([0,0]).view(1,1,1,2)
# print(grid)
myres = nearestGridSample(input,grid)
res = F.grid_sample(input,grid, mode='nearest', align_corners=False)
print(myres == res)
print(torch.sum(abs(myres-res)))

    
    
