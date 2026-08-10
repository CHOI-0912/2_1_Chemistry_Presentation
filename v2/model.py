import torch
import torch.nn as nn

MAX_Z = 10  # H(1) ~ F(9), 0은 패딩


class SimpleNNP(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def to_RCS(self, x): #Relative Coordinate System
        '''
        [b, n, 3] -> [b, n, n, 3]
        [n, n] 부분에서 [i, j]는 i가 본 j의 상대좌표 입니다.
        [i, i] = [0,0,0]입니다
        '''
        res = x[:, None, :, :] - x[:, :, None, :]
        return res
    
    def forward(self, numbers, coords):
        """numbers (B,N), coords (B,N,3) → 에너지 (B,)"""
        rel_coords = self.to_RCS(coords)

        T = rel_coords.size(-1)
        mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=rel_coords.device))
        rel_sq_inverse = 1/ ((rel_coords ** 2).masked_fill(~mask, 0))

        print(rel_sq_inverse)




        
        return (1, 2, 3, ...)  # (B,)

    def energy_and_forces(self, numbers, coords, create_graph=False):
        """E 와 F = -∂E/∂r"""
        coords = coords.requires_grad_(True)
        energy = self.forward(numbers, coords)
        (grad,) = torch.autograd.grad(energy.sum(), coords, create_graph=create_graph)
        return energy, -grad



rel_coords = torch.rand((2, 3, 3))
T = rel_coords.size(-1)
mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=rel_coords.device), diagonal=0)
rel_sq_inverse = (((1/rel_coords) ** 2).masked_fill(~mask, 0))

print(rel_sq_inverse)