import torch
import torch.nn as nn
import torch.nn.functional as F
import constants as con
from einops import rearrange

class SwiGLUFFN(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.w_up   = nn.Linear(dim, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))

class SimpleModel(nn.Module):
    def __init__(self, num_atom_whole, atten_heads, atten_dim, inner_dim, number_propo):
        '''
        Simple 한 형태의 에너지 예측 모델

        HF 방식에서 양성자간 에너지는 물리적으로 계산
        num_atom: 현재 배치의 원자 갯수
        atten_dim: 어텐션 차원 (atten_heads 의 배수)
        inner_dim: 내부 차원 (atten_heads 의 배수)
        num_atom_whole: 원자 번호의 최댓값(현재 배치가 아닌 고려해야할 총 원자 번호)
        number_propo: 처음 Attention에서 몇개의 head가 원자번호에 비례하는 것을 처리할지
        '''
        super().__init__()
        assert atten_dim % atten_heads == 0, "atten_dim은 atten_heads의 배수여야 함"
        assert inner_dim % atten_heads == 0, "inner_dim은 atten_heads의 배수여야 함"
        assert 0 < number_propo < atten_heads, "number_propo는 1 이상 atten_heads 미만 (나머지 head가 임베딩 담당)"
        #attention 2x2해서 4개
        self.atten_heads = atten_heads
        self.number_propo = number_propo
        self.head_qk_dim = atten_dim // atten_heads #head당 QK 차원
        self.head_v_dim = inner_dim // atten_heads #head당 V 차원
        self.log_self_distance = nn.Parameter(torch.tensor(0.0)) # log(1.0) = 0 # 자기 자신 attention의 learnable bias, 어떤 실수든 유효

        # 비례 head 몫만 선형으로 만든다. QK는 Q/K로 chunk하니 ×2, V는 ×2 불필요하고 차원 기준도 head_v_dim
        self.A1_WQK = nn.Linear(1, number_propo*self.head_qk_dim*2, dtype=torch.float32)
        self.A1_WV = nn.Linear(1, number_propo*self.head_v_dim, dtype=torch.float32)
        self.A1_SwiGLUFFN = SwiGLUFFN(inner_dim, inner_dim*4)

        self.A2_WQK = nn.Linear(inner_dim, atten_dim*2, dtype=torch.float32)
        self.A2_WV = nn.Linear(inner_dim, inner_dim, dtype=torch.float32)
        self.A2_SwiGLUFFN = SwiGLUFFN(inner_dim, inner_dim*4)

        self.A3_WQK = nn.Linear(inner_dim, atten_dim*2, dtype=torch.float32)
        self.A3_WV = nn.Linear(inner_dim, inner_dim, dtype=torch.float32)
        self.A3_SwiGLUFFN = SwiGLUFFN(inner_dim, inner_dim*4)

        self.energy_head = nn.Linear(inner_dim, 1) # 원자별 에너지 기여 -> 합산

        # 비비례 head 몫은 임베딩. Q/K는 atten_dim 기준, V는 inner_dim 기준으로 남은 차원을 채움
        self.embq = nn.Embedding(num_atom_whole+1, atten_dim-number_propo*self.head_qk_dim)
        self.embk = nn.Embedding(num_atom_whole+1, atten_dim-number_propo*self.head_qk_dim)
        self.embv = nn.Embedding(num_atom_whole+1, inner_dim-number_propo*self.head_v_dim)

        # 고립원자 총에너지 테이블 (Hartree). 매 forward마다 tensor 만들면 낭비(GPU면 매번 복사)라 버퍼로 1회 등록
        # 값이 수천 Ha라 float32(유효 ~7자리)면 원자당 ~1e-4 Ha가 뭉개짐 -> float64 유지
        self.register_buffer("Etot_table", torch.tensor(con.Etot, dtype=torch.float64)) # Etot_table[0] = 0 (패딩)

    def to_RCS(self, x): #Relative Coordinate System
        '''
        [b, n, 3] -> [b, n, n, 3]
        [n, n] 부분에서 [i, j]는 i가 본 j의 상대좌표 입니다.
        [i, i] = [0,0,0]입니다
        '''
        res = x[:, None, :, :] - x[:, :, None, :]
        return res

    def make_Etot(self, numbers:torch.Tensor):
        '''
        numbers (B,N) → '무한히 떨어진 고립원자'들의 총에너지 합 (B,)
        '''
        E = self.Etot_table[numbers].sum(-1) # 패딩(Z=0)은 테이블 값이 0이라 기여 없음
        return E

    def forward(self, numbers, coords):
        """
        numbers (B,N), coords (B,N,3) → 에너지 (B,)
        고립원자 몫(원자 자체 에너지): Etot 테이블 (큰 값 담당)
        분자 형성에 따른 나머지(결합/전자 재배치): 신경망 (미세한 변화 담당)
        원자핵 운동에너지: 보른-오펜하이머 근사에 의해 무시

        핵간쿨롱(E_nucnuc)을 명시항으로 넣지 않는 이유(2026-08-10 QM9x 3000분자 실측):
        결합 거리에서 핵간 반발(+수백 Ha)은 전자 안정화와 거의 상쇄되는데, 반발만 명시하면
        상쇄 반대편(-수백 Ha)을 신경망이 통째로 배워야 함 → NN 타깃 표준편차 78 Ha.
        빼면 타깃이 결합에너지(σ=0.7 Ha)로 줄어 100배 쉬워짐. 단거리 반발벽이 필요해지면
        bare 쿨롱이 아니라 차폐(screened/ZBL)형으로 넣을 것.


        ###################################################################################################################
        추가 아이디어- 전자 위치도 정의해 최적화 대상으로 바라보기 -> 전자-전자, 원자핵-전자 에너지 계산 가능
        ###################################################################################################################
        """
        T = numbers.size(-1)
        eye = torch.eye(T, device=coords.device, dtype=torch.bool)  # (N, N)
        padding = (numbers == 0) # (B, N) # 패딩 원자 위치

        rel_coords = self.to_RCS(coords) #(b, n, n, 3)
        rel_sq = (rel_coords**2).sum(-1) # (B, N, N) # 대각성분 0
        # 거리 0인 쌍은 대각만이 아님: 패딩 원자들이 전부 (0,0,0)에 겹쳐 있어 (패딩,패딩)·(원점의 실원자,패딩)도 0
        # 그대로 두면 쿨롱 1/0=inf(전하곱 0과 만나 0*inf=NaN), log(0)=-inf, backward의 sqrt'(0)=inf 세 곳에서 NaN
        # 이 쌍들의 거리값은 어디서도 안 쓰임(전하곱 0, 어텐션 -inf 차단, readout 마스킹) → 아무 양수나 OK, 1 사용
        pad_pair = padding[:, :, None] | padding[:, None, :] # (B, N, N) # 패딩이 한쪽이라도 낀 쌍
        rel_sq = rel_sq.masked_fill(eye | pad_pair, 1.0) # 반드시 sqrt 전 (sqrt 후에 막으면 backward에서 0*inf=NaN 재발)
        rel_distances = torch.sqrt(rel_sq) # (B, N, N) # 처리된 쌍은 1

        #학습 대상이 아닌 물리적으로 얻어낸 에너지
        Etot_energy = self.make_Etot(numbers) # (B, ) # '무한히 떨어진 고립원자'들의 총에너지
        ###################################################################################################################
        # Eenuc(원자핵-전자만) 대신 Etot을 쓰는 이유: Eenuc만 빼면 전자 운동E + 전자-전자 반발이 전부 신경망 몫이 됨
        # (물 기준 잔차 약 +94 Ha). Etot이면 신경망 몫 = 결합(원자화)에너지 몇 Ha 수준으로 줄고,
        # 원자들이 무한히 멀어지는 극한에서 잔차 → 0 이라 물리적으로도 깔끔함
        ###################################################################################################################

        #그 나머지

        # attention score를 거리/1로 나누고 정규화. 아래 코드는 log 이용해서 간단하게 구현한거
        # 대각은 log(거리) 대신 학습되는 log 값 자체를 사용 (양수 제약 불필요, log(0) 문제 원천 차단)
        log_inv_distance = torch.where(eye, self.log_self_distance, torch.log(rel_distances)) # (B, N, N)


        #A1 # 원자번호가 일부 의미를 가짐 + 거리 분수함수 가중 attention

        #Attention
        # flex_attention을 안 쓰는 이유(2026-08-10 실측): flex_attention_backward의 2차 미분이 미구현이라
        # 힘 손실(create_graph=True) 학습이 CUDA에서도 불가, CPU는 1차 backward부터 미지원.
        # 아래 수동 구현은 flex + score_mod와 수학적으로 동일 (기본 스케일 1/sqrt(head_qk_dim) 포함).
        scale = self.head_qk_dim ** -0.5
        blocked = padding[:, None, None, :] & ~eye[None, None, :, :] # (B,1,N,N) # 패딩 key 차단, 자기 자신은 남김 (전부 -inf면 softmax가 NaN)
        bias = log_inv_distance[:, None, :, :] # (B,1,N,N) # score에서 log(거리)를 빼면 = 가중치를 거리로 나누는 것

        def attend(q, k, v):
            scores = torch.einsum("bhqd,bhkd->bhqk", q, k) * scale - bias # (B,H,N,N)
            scores = scores.masked_fill(blocked, float("-inf"))
            return torch.einsum("bhqk,bhkd->bhqd", scores.softmax(-1), v) # (B,H,N,head_v_dim)

        # 비례 head: 원자번호 스칼라의 선형변환. 출력이 number_propo개 head 분량뿐이라 H=number_propo로 나눠야 함
        qk = rearrange(self.A1_WQK(numbers.float().unsqueeze(-1)), "B N (H D) -> B H N D", H=self.number_propo) # (B, number_propo, N, head_qk_dim*2)
        v_propo = rearrange(self.A1_WV(numbers.float().unsqueeze(-1)), "B N (H D) -> B H N D", H=self.number_propo) # (B, number_propo, N, head_v_dim)
        q_propo, k_propo = qk.chunk(2, dim=-1) # 각 (B, number_propo, N, head_qk_dim)
        # 비비례 head: 임베딩이 Q/K/V 담당. 같은 head 모양으로 쪼갬
        H_non = self.atten_heads - self.number_propo
        q_non_propo = rearrange(self.embq(numbers), "B N (H D) -> B H N D", H=H_non) # (B, H_non, N, head_qk_dim)
        k_non_propo = rearrange(self.embk(numbers), "B N (H D) -> B H N D", H=H_non)
        v_non_propo = rearrange(self.embv(numbers), "B N (H D) -> B H N D", H=H_non) # (B, H_non, N, head_v_dim)
        # 결합은 feature 축이 아니라 head 축(dim=1) — head마다 역할이 다르다는 설계이므로
        q = torch.cat([q_propo, q_non_propo], dim=1) # (B, atten_heads, N, head_qk_dim)
        k = torch.cat([k_propo, k_non_propo], dim=1)
        v = torch.cat([v_propo, v_non_propo], dim=1) # (B, atten_heads, N, head_v_dim)
        out = attend(q, k, v) # (B, H, N, head_v_dim)
        out = rearrange(out, "B H N D -> B N (H D)") # (B, N, inner_dim)
        out = self.A1_SwiGLUFFN(out)

        #A2, A3 #거리 분수함수 가중 attention
        qk = rearrange(self.A2_WQK(out), "B N (H D) -> B H N D", H=self.atten_heads) # (B, H, N, head_qk_dim*2)
        v = rearrange(self.A2_WV(out), "B N (H D) -> B H N D", H=self.atten_heads) # (B, H, N, head_v_dim)
        q, k = qk.chunk(2, dim=-1) # 각 (B, H, N, head_qk_dim)
        out = attend(q, k, v) # (B, H, N, head_v_dim)
        out = rearrange(out, "B H N D -> B N (H D)") # (B, N, inner_dim)
        out = self.A2_SwiGLUFFN(out)

        qk = rearrange(self.A3_WQK(out), "B N (H D) -> B H N D", H=self.atten_heads) # (B, H, N, head_qk_dim*2)
        v = rearrange(self.A3_WV(out), "B N (H D) -> B H N D", H=self.atten_heads) # (B, H, N, head_v_dim)
        q, k = qk.chunk(2, dim=-1) # 각 (B, H, N, head_qk_dim)
        out = attend(q, k, v) # (B, H, N, head_v_dim)
        out = rearrange(out, "B H N D -> B N (H D)") # (B, N, inner_dim)
        out = self.A3_SwiGLUFFN(out)

        # 에너지 구하기
        atom_E = self.energy_head(out).squeeze(-1) # (B, N) # 원자별 에너지 기여
        atom_E = atom_E.masked_fill(padding, 0.0) # 패딩 원자는 에너지 기여 제거
        nn_energy = atom_E.sum(-1) # (B, )

        E = Etot_energy + nn_energy # (B, ) # Etot이 float64라 전체가 float64로 승격됨 (테이블 정밀도 유지)
        return E # (B,)

    def energy_and_forces(self, numbers, coords, create_graph=False):
        """E 와 F = -∂E/∂r"""
        coords = coords.requires_grad_(True)
        energy = self.forward(numbers, coords)
        (grad,) = torch.autograd.grad(energy.sum(), coords, create_graph=create_graph)
        return energy, -grad
