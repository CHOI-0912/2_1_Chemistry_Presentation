from preprocess import iter_molecules
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from einops import rearrange

class Chemical_Model(nn.Module):
    def __init__(self, dim, rel_dim, num_atoms):
        '''
        1차적으로 주변 관계를 이용해서 원자를 update
        2차적으로 거리, 관계 원자간 관계를 이용해서 Energy 계산

        1차 update시 0번 원소를 기준점(물리학 적으로 말 됨. 전부 상대적 위치관계니까)
        이거 기준으로 각도, 거리 정보와 원자정보로 attention이나 -거리 softmax를 이용해서 update
        dim을 2 부분으로 나눠서 각도따라 달라짐 / 안달라짐 두 파트로
        '''
        super().__init__()
        self.dim = dim
        self.emb = nn.Embedding(4, self.dim)

        self.rel_emb = nn.Embedding(num_atoms**2, rel_dim) #3.4옹스트롬 기준 원자 쌍별 상호작용
        # 두 원자간 상호작용을 모사한다. 많은 것들이 섞여 있어서 힘은 아니다 ex: 힘, 전자 끌림, 기타...

        # 서은희 선생님 PPT에 따라 두 원자간 상호작용은 r^-6으로 하기로 함. 수십 nm 거리에서는 지연효과 떄문에 r^-7이 되기도 하고 너무 가까우면 r^-12가 되기도 하지만 너무 극단적인 케이스 들이기도 하고 효율성을 지키며 구현하기 어려울 것 같아서 r^-6으로 함
        # 거리 기준은 3.4임. 표준편차가 평균의 50%로 높긴 하지만 평균과 Q50이 비슷해서 그냥 그 사이에 의미있는 숫자인 3.4로함 (나는야 34기^^)

        self.bigeps = 1e-5
        self.eps = 1e-7

        #St1
        
        self.rms = nn.RMSNorm(rel_dim)

        self.layer_nri1 = nn.Sequential( # Not rotationally invariant
            nn.Linear(rel_dim, rel_dim * 4),
            nn.GELU(),
            nn.Linear(rel_dim*4, rel_dim),
            nn.GELU(),
            )
        
        self.layer_nri2 = nn.Sequential( # Not rotationally invariant
            nn.RMSNorm(rel_dim),
            nn.Linear(rel_dim, rel_dim * 4),
            nn.GELU(),
            nn.Linear(rel_dim*4, rel_dim),
            nn.GELU(),
            )
        
        #St2
        st2_dim = dim + rel_dim

        self.layer_ri1 = nn.Sequential( # rotationally invariant
            nn.RMSNorm(st2_dim),
            nn.Linear(st2_dim, st2_dim * 4),
            nn.GELU(),
            nn.Linear(st2_dim*4, st2_dim),
            nn.GELU(),
            )
        
        self.layer_ri2 = nn.Sequential( # rotationally invariant
            nn.RMSNorm(st2_dim),
            nn.Linear(st2_dim, st2_dim * 4),
            nn.GELU(),
            nn.Linear(st2_dim*4, st2_dim),
            nn.GELU(),
            )
        
        # Step 3
        self.ENERGY = nn.Linear(st2_dim, 1) # 이 친구가 구하는 값이 에너지 라고 보기


    
    def to_RCS(self, x): #Relative Coordinate System
        '''
        [b, n, 3] -> [b, n, n, 3]
        [n, n] 부분에서 [i, j]는 i가 본 j의 상대좌표 입니다.
        [i, i] = [0,0,0]입니다
        '''
        res = x[:, None, :, :] - x[:, :, None, :]
        return res

    def make_relative_btw_atom_not_consider_length(self, molecules):
        temp1 = torch.max(molecules[:, :, None], molecules[:, None, :])
        temp2 = torch.min(molecules[:, :, None], molecules[:, None, :])

        relative_btw_atom = (temp1*(temp1+1))//2 + temp2

        return relative_btw_atom

    def make_relative_btw_atom_consider_length(self, molecules, rec_distances):
        mask = torch.eye(rec_distances.size(-1), device=rec_distances.device, dtype=torch.bool)
        rec_distances = torch.where(mask, float('inf'), rec_distances) # 자기자신 거리 inf로 해서 역수 취하면 0 -> 자기자신 상호작용 무시

        rec_distances = 3.4 / rec_distances # 스케일링 & r^(-6)에서 - 적용
        rec_distances = torch.pow(rec_distances, 6)

        #기준 거리에서 분자간 상호작용 크기 구하기 (함수 이름 뜻이 이거임)
        relative_btw_atom = self.make_relative_btw_atom_not_consider_length(molecules) #[b, n, n] 대각 성분 무시
        relative_btw_atom = self.rel_emb(relative_btw_atom) #[b, n, n, rel_dim] 대각 성분 무시

        #합쳐서 분자간 상호작용 구하기
        relative_btw_atom = relative_btw_atom * rec_distances.unsqueeze(-1)

        return relative_btw_atom

    def Step1(self, molecules, coords):
        # r*(-6) 구하기 Step1-1
        rcs = self.to_RCS(coords) # [b, n, n, 3] #대각성분 0

        rec_distances = torch.pow(torch.matmul(rcs[:, :, :, None, :], rcs[:, :, :, :, None]).squeeze(-1).squeeze(-1)+ self.bigeps, 0.5)
        # ^ [b, n, n] # 대각 성분 0

        #거리를 고려한 분자간 상호작용 크기 구하기 Step1-2 (함수 이름 뜻이 이거임)
        relative_btw_atom = self.make_relative_btw_atom_consider_length(molecules, rec_distances)
        # ^[b, n, n, rel_dim]

        relative_btw_atom = self.rms(relative_btw_atom)
        relative_btw_atom = relative_btw_atom + self.layer_nri1(relative_btw_atom)
        relative_btw_atom = relative_btw_atom + self.layer_nri2(relative_btw_atom)
        # ^[b, n, n, rel_dim]

        #힘에 대한 방향 벡터 제작 Step1-3
        rcs_unit = rcs / rec_distances[: ,:, :, None]
        # ^[b, n, n, 3]

        #회전 시키기 Step 1 마무리
        relative_btw_atom = relative_btw_atom[:, :, :, :, None] * rcs_unit[:, :, :, None, :]
    
        return relative_btw_atom

    def Step2(self, x):
        '''
        걍 대충 mlp 짜그리기
        '''
        x = x + self.layer_ri1(x)
        x = x + self.layer_ri2(x)

        return x
 

    def forward(self, molecules, coords):
        '''
        b: batch_size
        n: 화학 분자를 이루는 원자의 갯수
        3: xyz좌표
        molecules: [b, n]
        coords: [b, n , 3]
        '''
        relative_btw_atom = self.Step1(molecules, coords) #각 단위거리에 정의된 물리학적 영향에 거리, 방향을 고려한 벡터
        # ^[b, n, n, rel_dim, 3]

        relative_btw_atom = rearrange(relative_btw_atom, "b n m r t -> b n r t m") # n = m
        # ^[b, n, 3, rel_dim, n]

        relative_btw_atom = torch.sum(relative_btw_atom, dim=-1) # 각 물리학적 영향을 '합한다' 즉, 물리학적 영향 벡터들을 합한다
        # ^[b, n, rel_dim, 3]

        relative_btw_atom = torch.pow(torch.matmul(relative_btw_atom[:, :, :, None, :], relative_btw_atom[:, :, :, :, None]).squeeze(-1).squeeze(-1)+ self.bigeps, 0.5)
        # ^[b, n, rel_dim]

        # 이제 이럼으로서 회전 동등하게 학습 가능함
        relative_btw_atom = torch.cat([self.emb(molecules), relative_btw_atom], dim=-1)
        # ^[b, n , dim+rel_dim]
        
        relative_btw_atom = self.Step2(relative_btw_atom)
        ENERGY = self.ENERGY(relative_btw_atom).squeeze(-1)
        return torch.sum(ENERGY, dim = -1)

if __name__ == '__main__':
    mol = next(iter_molecules())
    print(mol.keys())

    print(mol["formula"])
    print(mol["atomic_numbers"].shape)
    print(mol["coordinates"].shape)
    print(mol["energy"].shape)

    #데이터셋 짰다 치고


    mols = torch.randint(0, 4, [10, 5]) #b, n
    coords = torch.randn([10, 5, 3]) #[b, n, 3]


    model = Chemical_Model(dim=128, rel_dim=128, num_atoms=4) #rel_dim 나중에 내부적으로 크기 3배됨

    print(model(mols, coords).shape)