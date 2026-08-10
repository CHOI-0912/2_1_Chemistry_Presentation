import torch
import torch.nn as nn
import torch.nn.functional as F
import constants as con
from torch.nn.attention.flex_attention import flex_attention
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

    def atomic_nucleus_energy(self, numbers:torch.Tensor, rel_distances:torch.Tensor):
        '''
        numbers (B,N), coords (B,N,3) → 원자핵간 "쿨롱" 에너지 (B,)
        '''
        # q_1 * q_2 계산
        numbers_matrix = numbers[:, None, :] * numbers[:, :, None] # (B, N, N)

        T = rel_distances.size(-1)
        eye = torch.eye(T, device=rel_distances.device, dtype=torch.bool)  # (N, N)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=rel_distances.device), diagonal=1) # (N, N) #하삼각 의미 없음 (중복 계산)

        # mask에서 rel_distances의 원래 대각성분이 0이라 추후 나누기에서 문제가 생김. eye로 대각성분을 1로 넣은거
        rel_inverse = ((1/rel_distances.masked_fill(eye, 1.0)).masked_fill(~mask, 0)) # (B, N, N) # 대각 + 하삼각으로 서로 다른 쌍 중복없이 고려

        coefficients = numbers_matrix * rel_inverse # (B, N, N) -> (B, N) # 쿨롱 퍼텐셜의 달라지는 부분 (계수)
        E = coefficients.sum(-1).sum(-1) * con.E2_OVER_4PI_EPS0_HARTREE_ANG # (B, )
        return E # (B, )

    def make_Etot(self, numbers:torch.Tensor):
        '''
        numbers (B,N) → '무한히 떨어진 고립원자'들의 총에너지 합 (B,)
        '''
        E = self.Etot_table[numbers].sum(-1) # 패딩(Z=0)은 테이블 값이 0이라 기여 없음
        return E

    def forward(self, numbers, coords):
        """
        numbers (B,N), coords (B,N,3) → 에너지 (B,)
        하트리-폭 기반의 에너지 계산을 참고하여 제작
        원자핵-원자핵: atomic_nucleus_energy (쿨롱, 물리 계산)
        고립원자 몫(전자 운동 + 원자핵-전자 + 전자-전자 + XC): Etot 테이블 (큰 값 담당)
        분자 형성에 따른 나머지(결합/전자 재배치): 신경망 (미세한 변화 담당)
        원자핵 운동에너지: 보른-오펜하이머 근사에 의해 무시


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

        #학습 대상이 아닌 물리적으로 얻어낸 에너지 단, 아래 에너지는 좌표정보가 쓰여 미분 가능하면 좋음
        atomic_nucleus_energy = self.atomic_nucleus_energy(numbers, rel_distances) # (B, ) # 원자핵간 쿨롱 에너지

        #학습 대상이 아닌 물리적으로 얻어낸 에너지
        Etot_energy = self.make_Etot(numbers) # (B, ) # '무한히 떨어진 고립원자'들의 총에너지
        ###################################################################################################################
        # Eenuc(원자핵-전자만) 대신 Etot을 쓰는 이유: Eenuc만 빼면 전자 운동E + 전자-전자 반발이 전부 신경망 몫이 됨
        # (물 기준 잔차 약 +94 Ha). Etot이면 잔차 = 원자화E - 핵간쿨롱 상쇄분 ≈ -10 Ha 수준으로 줄고,
        # 원자들이 무한히 멀어지는 극한에서 핵간쿨롱 → 0, 잔차 → 0 이라 물리적으로도 깔끔함
        ###################################################################################################################

        #그 나머지

        # attention score를 거리/1로 나누고 정규화. 아래 코드는 log 이용해서 간단하게 구현한거
        # 대각은 log(거리) 대신 학습되는 log 값 자체를 사용 (양수 제약 불필요, log(0) 문제 원천 차단)
        log_inv_distance = torch.where(eye, self.log_self_distance, torch.log(rel_distances)) # (B, N, N)


        #A1 # 원자번호가 일부 의미를 가짐 + 거리 분수함수 가중 attention

        def score_mod(score, b, h, q_idx, kv_idx):
            # score: 스칼라 score (q_idx번째 query와 kv_idx번째 key의 내적)
            # b: batch index, h: head index
            # 패딩 key는 차단하되 자기 자신은 남김 (전부 -inf면 softmax가 NaN)
            blocked = padding[b, kv_idx] & (q_idx != kv_idx)
            return torch.where(blocked, float("-inf"), score - log_inv_distance[b, q_idx, kv_idx])

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
        out = flex_attention(q, k, v, score_mod=score_mod) # (B, H, N, head_v_dim)
        out = rearrange(out, "B H N D -> B N (H D)") # (B, N, inner_dim)
        out = self.A1_SwiGLUFFN(out)

        #A2, A3 #거리 분수함수 가중 attention
        qk = rearrange(self.A2_WQK(out), "B N (H D) -> B H N D", H=self.atten_heads) # (B, H, N, head_qk_dim*2)
        v = rearrange(self.A2_WV(out), "B N (H D) -> B H N D", H=self.atten_heads) # (B, H, N, head_v_dim)
        q, k = qk.chunk(2, dim=-1) # 각 (B, H, N, head_qk_dim)
        out = flex_attention(q, k, v, score_mod=score_mod) # (B, H, N, head_v_dim)
        out = rearrange(out, "B H N D -> B N (H D)") # (B, N, inner_dim)
        out = self.A2_SwiGLUFFN(out)

        qk = rearrange(self.A3_WQK(out), "B N (H D) -> B H N D", H=self.atten_heads) # (B, H, N, head_qk_dim*2)
        v = rearrange(self.A3_WV(out), "B N (H D) -> B H N D", H=self.atten_heads) # (B, H, N, head_v_dim)
        q, k = qk.chunk(2, dim=-1) # 각 (B, H, N, head_qk_dim)
        out = flex_attention(q, k, v, score_mod=score_mod) # (B, H, N, head_v_dim)
        out = rearrange(out, "B H N D -> B N (H D)") # (B, N, inner_dim)
        out = self.A3_SwiGLUFFN(out)

        # 에너지 구하기
        atom_E = self.energy_head(out).squeeze(-1) # (B, N) # 원자별 에너지 기여
        atom_E = atom_E.masked_fill(padding, 0.0) # 패딩 원자는 에너지 기여 제거
        nn_energy = atom_E.sum(-1) # (B, )

        E = atomic_nucleus_energy + Etot_energy + nn_energy # (B, ) # Etot이 float64라 전체가 float64로 승격됨 (테이블 정밀도 유지)
        return E # (B,)

    def energy_and_forces(self, numbers, coords, create_graph=False):
        """E 와 F = -∂E/∂r"""
        coords = coords.requires_grad_(True)
        energy = self.forward(numbers, coords)
        (grad,) = torch.autograd.grad(energy.sum(), coords, create_graph=create_graph)
        return energy, -grad
