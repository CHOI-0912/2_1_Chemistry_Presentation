"""가장 단순한 신경망 퍼텐셜(NNP).

    1. 원자번호 → 임베딩 h_i
    2. 원자 쌍 거리 r_ij 를 가우시안 기저(RBF)로 펼친다
    3. 이웃의 h_j 를 거리 기반 필터로 가중합해 h_i 를 갱신한다 (× n_blocks)
    4. 원자별 에너지 e_i 를 뽑아 전부 더한 것이 분자의 에너지 E

힘은 따로 예측하지 않고 물리 정의 그대로 F_i = -∂E/∂r_i 를 autograd로 얻는다.

좌표 Å, 에너지 Hartree (v2/dataloader.py와 동일). 원자번호 0 = 패딩.
"""

import torch
import torch.nn as nn

MAX_Z = 10  # H(1) ~ F(9), 0은 패딩


class SimpleNNP(nn.Module):
    def __init__(self, hidden=64, n_rbf=32, cutoff=5.0, n_blocks=3):
        super().__init__()
        self.cutoff = cutoff
        self.gamma = (n_rbf / cutoff) ** 2

        self.embedding = nn.Embedding(MAX_Z, hidden, padding_idx=0)
        self.register_buffer("centers", torch.linspace(0.0, cutoff, n_rbf))

        self.filters = nn.ModuleList(
            nn.Sequential(nn.Linear(n_rbf, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
            for _ in range(n_blocks)
        )
        self.updates = nn.ModuleList(
            nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
            for _ in range(n_blocks)
        )
        self.readout = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.SiLU(), nn.Linear(hidden // 2, 1)
        )

    def forward(self, numbers, coords):
        """numbers (B,N), coords (B,N,3) → 에너지 (B,)"""
        mask = numbers > 0

        diff = coords[:, :, None, :] - coords[:, None, :, :]
        dist = torch.sqrt((diff**2).sum(-1) + 1e-12)  # (B,N,N)

        pair = mask[:, :, None] & mask[:, None, :]
        pair = pair & ~torch.eye(numbers.shape[1], dtype=torch.bool, device=numbers.device)
        pair = pair & (dist < self.cutoff)

        rbf = torch.exp(-self.gamma * (dist[..., None] - self.centers) ** 2)
        fcut = 0.5 * (torch.cos(torch.pi * dist / self.cutoff) + 1.0)
        gate = (fcut * pair)[..., None]  # 컷오프 밖·패딩 쌍은 0

        h = self.embedding(numbers)
        for filt, upd in zip(self.filters, self.updates):
            w = filt(rbf) * gate
            msg = (w * h[:, None, :, :]).sum(dim=2)  # 이웃 j에 대한 합
            h = h + upd(msg)

        e_atom = self.readout(h).squeeze(-1)  # (B,N)
        return (e_atom * mask).sum(dim=1)  # (B,)

    def energy_and_forces(self, numbers, coords, create_graph=False):
        """E 와 F = -∂E/∂r"""
        coords = coords.requires_grad_(True)
        energy = self.forward(numbers, coords)
        (grad,) = torch.autograd.grad(energy.sum(), coords, create_graph=create_graph)
        return energy, -grad
