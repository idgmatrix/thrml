import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import List, Dict, Tuple

# ============================================================================
# 1. QUBO 구성 (4x4 스도쿠)
# ============================================================================

def build_4x4_sudoku_qubo(puzzle: List[List[int]]) -> Dict[Tuple[int, int], float]:
    """4x4 스도쿠를 QUBO로 변환"""
    N = 4
    penalty = 10.0
    qubo = {}
    
    def var_index(i, j, k):
        """(row, col, digit) → flat index"""
        return i * N * N + j * N + k
    
    # Constraint 1: 각 셀에 정확히 하나의 숫자
    for i in range(N):
        for j in range(N):
            for k1 in range(N):
                v1 = var_index(i, j, k1)
                qubo[(v1, v1)] = qubo.get((v1, v1), 0.0) - penalty
                for k2 in range(k1 + 1, N):
                    v2 = var_index(i, j, k2)
                    key = (min(v1, v2), max(v1, v2))
                    qubo[key] = qubo.get(key, 0.0) + 2 * penalty
    
    # Constraint 2: 각 행에 각 숫자가 정확히 한 번
    for i in range(N):
        for k in range(N):
            for j1 in range(N):
                v1 = var_index(i, j1, k)
                qubo[(v1, v1)] = qubo.get((v1, v1), 0.0) - penalty
                for j2 in range(j1 + 1, N):
                    v2 = var_index(i, j2, k)
                    key = (min(v1, v2), max(v1, v2))
                    qubo[key] = qubo.get(key, 0.0) + 2 * penalty
    
    # Constraint 3: 각 열에 각 숫자가 정확히 한 번
    for j in range(N):
        for k in range(N):
            for i1 in range(N):
                v1 = var_index(i1, j, k)
                qubo[(v1, v1)] = qubo.get((v1, v1), 0.0) - penalty
                for i2 in range(i1 + 1, N):
                    v2 = var_index(i2, j, k)
                    key = (min(v1, v2), max(v1, v2))
                    qubo[key] = qubo.get(key, 0.0) + 2 * penalty
    
    # Constraint 4: 각 2x2 박스에 각 숫자가 정확히 한 번
    for box_r in range(2):
        for box_c in range(2):
            for k in range(N):
                cells = []
                for dr in range(2):
                    for dc in range(2):
                        i = box_r * 2 + dr
                        j = box_c * 2 + dc
                        cells.append(var_index(i, j, k))
                
                for idx1 in range(len(cells)):
                    v1 = cells[idx1]
                    qubo[(v1, v1)] = qubo.get((v1, v1), 0.0) - penalty
                    for idx2 in range(idx1 + 1, len(cells)):
                        v2 = cells[idx2]
                        key = (min(v1, v2), max(v1, v2))
                        qubo[key] = qubo.get(key, 0.0) + 2 * penalty
    
    # Constraint 5: 주어진 clue 고정 (강한 페널티)
    for i in range(N):
        for j in range(N):
            if puzzle[i][j] != 0:
                digit = puzzle[i][j] - 1  # 0-indexed
                v_correct = var_index(i, j, digit)
                # 올바른 변수는 강하게 선호
                qubo[(v_correct, v_correct)] = qubo.get((v_correct, v_correct), 0.0) - 50.0
                
                # 다른 숫자들은 큰 페널티
                for k in range(N):
                    if k != digit:
                        v_wrong = var_index(i, j, k)
                        qubo[(v_wrong, v_wrong)] = qubo.get((v_wrong, v_wrong), 0.0) + 50.0
    
    return qubo


def qubo_to_ising(qubo: Dict[Tuple[int, int], float]) -> Tuple[jnp.ndarray, Dict, float]:
    """QUBO {0,1} → Ising {-1,+1} 변환"""
    if not qubo:
        return jnp.array([]), {}, 0.0
    
    max_var = max(max(i, j) for i, j in qubo.keys())
    N = max_var + 1
    
    h = np.zeros(N)
    J_dict = {}
    const = 0.0
    
    for (i, j), coeff in qubo.items():
        if i == j:
            h[i] += coeff / 2.0
            const += coeff / 2.0
        else:
            J_dict[(i, j)] = coeff / 4.0
            h[i] += coeff / 4.0
            h[j] += coeff / 4.0
            const += coeff / 4.0
    
    return jnp.array(h), J_dict, const


# ============================================================================
# 2. 개선된 Gibbs Sampling
# ============================================================================

def gibbs_sampling_step(state: jnp.ndarray, h: jnp.ndarray, J_dict: Dict, 
                        beta: float, key: jax.random.PRNGKey) -> jnp.ndarray:
    """한 스텝의 Gibbs sampling - 랜덤 순서로 업데이트"""
    N = len(state)
    
    # 랜덤 순서로 변수 업데이트
    key, subkey = jax.random.split(key)
    indices = jax.random.permutation(subkey, N)
    
    for idx in indices:
        i = int(idx)
        
        # 이웃의 영향 계산
        field = float(h[i])
        for (a, b), J_val in J_dict.items():
            if a == i:
                field += float(J_val * state[b])
            elif b == i:
                field += float(J_val * state[a])
        
        # Boltzmann 확률로 spin 결정
        prob_plus = 1.0 / (1.0 + jnp.exp(-2.0 * beta * field))
        key, subkey = jax.random.split(key)
        new_spin = jnp.where(jax.random.uniform(subkey) < prob_plus, 1, -1)
        state = state.at[i].set(new_spin)
    
    return state


def decode_4x4_board(spin_state: jnp.ndarray) -> np.ndarray:
    """Spin 상태 → 4x4 보드 (개선된 디코딩)"""
    N = 4
    board = np.zeros((N, N), dtype=int)
    
    for i in range(N):
        for j in range(N):
            # 각 셀에서 가장 높은 확률의 숫자 선택
            max_k = -1
            max_val = -1
            for k in range(N):
                idx = i * N * N + j * N + k
                if idx < len(spin_state):
                    # spin이 +1이면 해당 숫자 선택
                    if spin_state[idx] > 0 and spin_state[idx] > max_val:
                        max_val = spin_state[idx]
                        max_k = k
            
            if max_k >= 0:
                board[i, j] = max_k + 1
    
    return board


def calculate_energy(state: jnp.ndarray, h: jnp.ndarray, J_dict: Dict) -> float:
    """현재 상태의 에너지 계산"""
    energy = -jnp.dot(h, state)
    for (i, j), J_val in J_dict.items():
        energy -= J_val * state[i] * state[j]
    return float(energy)


# ============================================================================
# 3. 실시간 애니메이션 (에너지 그래프 추가)
# ============================================================================

def solve_4x4_sudoku_animated(puzzle: List[List[int]], 
                               beta: float = 1.5,
                               n_steps: int = 500,
                               seed: int = 42):
    """4x4 스도쿠를 풀면서 실시간 애니메이션"""
    
    # QUBO → Ising
    qubo = build_4x4_sudoku_qubo(puzzle)
    h, J_dict, _ = qubo_to_ising(qubo)
    
    # 초기 상태 (랜덤)
    key = jax.random.key(seed)
    N_vars = len(h)
    key, subkey = jax.random.split(key)
    state = jax.random.choice(subkey, jnp.array([-1, 1]), shape=(N_vars,))
    
    # 애니메이션 설정
    fig = plt.figure(figsize=(14, 5))
    gs = fig.add_gridspec(2, 3, height_ratios=[4, 1])
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[:, 2])
    
    # 왼쪽: 원본 퍼즐
    ax1.set_title("Original Puzzle", fontsize=14, fontweight='bold')
    ax1.set_xlim(0, 4)
    ax1.set_ylim(0, 4)
    ax1.set_aspect('equal')
    ax1.invert_yaxis()
    ax1.set_xticks([])
    ax1.set_yticks([])
    
    # 그리드 그리기
    for i in range(5):
        lw = 3 if i % 2 == 0 else 1
        ax1.axhline(i, color='black', linewidth=lw)
        ax1.axvline(i, color='black', linewidth=lw)
    
    # 원본 숫자 표시
    for i in range(4):
        for j in range(4):
            if puzzle[i][j] != 0:
                ax1.text(j + 0.5, i + 0.5, str(puzzle[i][j]), 
                        ha='center', va='center', fontsize=24, 
                        fontweight='bold', color='blue')
    
    # 중간: 샘플링 진행 상황
    ax2.set_title("Sampling Progress (Step 0)", fontsize=14, fontweight='bold')
    ax2.set_xlim(0, 4)
    ax2.set_ylim(0, 4)
    ax2.set_aspect('equal')
    ax2.invert_yaxis()
    ax2.set_xticks([])
    ax2.set_yticks([])
    
    # 그리드
    for i in range(5):
        lw = 3 if i % 2 == 0 else 1
        ax2.axhline(i, color='black', linewidth=lw)
        ax2.axvline(i, color='black', linewidth=lw)
    
    # 텍스트 객체 생성
    text_objects = []
    for i in range(4):
        row = []
        for j in range(4):
            txt = ax2.text(j + 0.5, i + 0.5, '', 
                          ha='center', va='center', fontsize=24)
            row.append(txt)
        text_objects.append(row)
    
    # 오른쪽: 에너지 그래프
    ax3.set_title("Energy over Time", fontsize=14, fontweight='bold')
    ax3.set_xlabel("Step")
    ax3.set_ylabel("Energy")
    ax3.grid(True, alpha=0.3)
    
    energy_history = [calculate_energy(state, h, J_dict)]
    line, = ax3.plot([0], energy_history, 'b-', linewidth=2)
    
    # 애니메이션 업데이트 함수
    states_history = [state]
    
    def update(frame):
        nonlocal state, key
        
        # Gibbs sampling 스텝
        key, subkey = jax.random.split(key)
        state = gibbs_sampling_step(state, h, J_dict, beta, subkey)
        states_history.append(state)
        
        # 에너지 계산
        energy = calculate_energy(state, h, J_dict)
        energy_history.append(energy)
        
        # 보드 디코딩
        board = decode_4x4_board(state)
        
        # 텍스트 업데이트
        for i in range(4):
            for j in range(4):
                val = board[i, j]
                if val != 0:
                    # 원본 clue는 파란색, 샘플링 결과는 검정색
                    if puzzle[i][j] != 0:
                        text_objects[i][j].set_text(str(val))
                        text_objects[i][j].set_color('blue')
                        text_objects[i][j].set_fontweight('bold')
                    else:
                        text_objects[i][j].set_text(str(val))
                        text_objects[i][j].set_color('red')
                        text_objects[i][j].set_fontweight('normal')
                else:
                    text_objects[i][j].set_text('?')
                    text_objects[i][j].set_color('gray')
        
        # 에너지 그래프 업데이트
        line.set_data(range(len(energy_history)), energy_history)
        ax3.relim()
        ax3.autoscale_view()
        
        ax2.set_title(f"Sampling Progress (Step {frame + 1}/{n_steps}, E={energy:.1f})", 
                     fontsize=14, fontweight='bold')
        
        return [item for row in text_objects for item in row] + [line]
    
    # 애니메이션 실행
    anim = animation.FuncAnimation(fig, update, frames=n_steps, 
                                   interval=100, blit=False, repeat=False)
    
    plt.tight_layout()
    plt.show()
    
    # 최종 결과 반환
    final_board = decode_4x4_board(states_history[-1])
    return final_board, states_history, energy_history


# ============================================================================
# 4. 테스트
# ============================================================================

# 4x4 스도쿠 예제 (0은 빈 칸)
puzzle_4x4 = [
    [1, 0, 0, 0],
    [0, 0, 2, 0],
    [0, 3, 0, 0],
    [0, 0, 0, 4]
]

print("4x4 스도쿠 풀이 시작!")
print("원본 퍼즐:")
for row in puzzle_4x4:
    print(row)

final_board, history, energies = solve_4x4_sudoku_animated(
    puzzle_4x4, 
    beta=1.0,  # 낮은 온도로 시작 (0.5~2.0)
    n_steps=500,
    seed=42
)

print("\n최종 결과:")
print(final_board)
print(f"\n최종 에너지: {energies[-1]:.2f}")