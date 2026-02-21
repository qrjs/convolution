from lib import *

# 将 DDR 中的 MK 布局矩阵转换为 L1 中的 K1MK0 布局
# matrix_mk: 原始矩阵
# tensor_m, tensor_k: 原始矩阵的完整维度
# start_m, start_k: 当前切片的起始坐标
# tile_m, tile_k: 当前切片的大小
def MK2K1MK0_DDR2L1(matrix_mk, tensor_m, tensor_k, start_m, start_k, tile_m, tile_k):
    # 计算 K 方向的分块数量
    K1 = ceil_div(tile_k, K0)
    # L1 中的 buffer 形状为 (K1, M, K0)
    matrix_k1mk0 = np.zeros((K1, tile_m, K0))
    for k1 in range(K1):
        for m in range(tile_m):
            for k0 in range(K0):
                # 计算 DDR 中的全局 K 坐标
                k = start_k + k1 * K0 + k0
                # 检查边界，防止越界并处理 padding
                if k < start_k + tile_k and k < tensor_k:
                    matrix_k1mk0[k1][m][k0] = matrix_mk[start_m + m][k]

    return matrix_k1mk0

# 执行分块后的矩阵乘法，包含偏置 (Bias)、累加 (Psum) 和 反量化 (Dequantization)
# m1n1m0n0: 输出结果所在的 L1 buffer
# m1k1m0k0, n1k1n0k0: 输入的分块矩阵 (L0 A/B buffer 内容)
# bias_n1n0, deq_n1n0: 偏置和反量化参数
# M1, N1, K1: 当前 Slice 的分块维度
# bias_en: 偏置使能信号
# psum_en: 部分积累加使能信号
# deq_en: 反量化使能信号
def matmul_m1k1m0k0_n1k1n0k0(m1n1m0n0, m1k1m0k0, n1k1n0k0, bias_n1n0, deq_n1n0, M1, N1, K1, bias_en, psum_en, deq_en):
    # 检查维度匹配
    assert(m1k1m0k0.shape[1] == n1k1n0k0.shape[1])
    assert(m1k1m0k0.shape[3] == n1k1n0k0.shape[3])
    
    for m1 in range(M1):
        for n1 in range(N1):
            # 执行核心计算：M0xK0 @ K0xN0 -> M0xN0
            temp = np.zeros((M0, N0))
            for k1 in range(K1):
                # left @ right.T: (M0, K0) @ (K0, N0) = (M0, N0)
                temp += m1k1m0k0[m1][k1] @ n1k1n0k0[n1][k1].T

            # 偏置使能：通常在第一次计算 K 维度时加偏置
            if(bias_en):
                for n0 in range(N0):
                    temp[:, n0] += bias_n1n0[n1][n0]

            # 部分积使能：如果不是第一块 K，则需要加到之前的计算结果上
            if(psum_en):
                m1n1m0n0[m1][n1] += temp
            else:
                m1n1m0n0[m1][n1] = temp
                
            # 反量化使能：通常在 K 维度全部累加完后进行
            if(deq_en):
                m1n1m0n0[m1][n1] *= deq_n1n0[n1]

    return m1n1m0n0 

def test_matmul():
    # 定义随机生成的 Tensor 各维度的总大小
    TENSOR_M = np.random.randint(3, 100)
    TENSOR_N = np.random.randint(3, 100)
    TENSOR_K = np.random.randint(3, 100)
    
    # 定义 Tile 大小 (L1 级缓存大小)
    TILE_M = np.random.randint(2, TENSOR_M)
    TILE_N = np.random.randint(2, TENSOR_N)
    TILE_K = np.random.randint(2, TENSOR_K)

    # 定义 Slice 大小 (L0 级寄存器/Buffer 大小，必须对齐到 M0, N0, K0)
    SLICE_M = ceil_align(np.random.randint(1, TILE_M), M0)
    SLICE_N = ceil_align(np.random.randint(1, TILE_N), N0)
    SLICE_K = ceil_align(np.random.randint(1, TILE_K), K0)
    SLICE_M1 = ceil_div(SLICE_M, M0)
    SLICE_N1 = ceil_div(SLICE_N, N0)

    # 打印层参数
    layer = {
        'M': TENSOR_M,
        'N': TENSOR_N,
        'K': TENSOR_K
    }
    print(layer)

    # 创建 DDR 模拟数据
    left = np.random.randint(-128, 127, size=(TENSOR_M, TENSOR_K))
    right = np.random.randint(-128, 127, size=(TENSOR_K, TENSOR_N))
    bias = np.random.randint(-128, 127, size=(TENSOR_N))
    deq = np.random.rand(TENSOR_N)
    right_nk = right.transpose() # 转换为 NK 方便处理
    result_mn = np.zeros((TENSOR_M, TENSOR_N))

    # 第一层循环：在 N 维度切分 Tile
    for tile_n_start_in_tensor in range(0, TENSOR_N, TILE_N):
        n_size_tile = min(TILE_N, TENSOR_N - tile_n_start_in_tensor)
        bias_n_tile = bias[tile_n_start_in_tensor:tile_n_start_in_tensor+n_size_tile]
        deq_n_tile = deq[tile_n_start_in_tensor:tile_n_start_in_tensor+n_size_tile]
        
        # 第二层循环：在 M 维度切分 Tile
        for tile_m_start_in_tensor in range(0, TENSOR_M, TILE_M):
            m_size_tile = min(TILE_M, TENSOR_M - tile_m_start_in_tensor)
            # PSB: Partial Sum Buffer (模拟 L1 中的输出 buffer)
            result_m2n2m1n1m0n0_psb = np.zeros((ceil_div(m_size_tile, SLICE_M)*ceil_div(n_size_tile, SLICE_N), SLICE_M1, SLICE_N1, M0, N0))
            
            # 第三层循环：在 K 维度切分 Tile
            for tile_k_start_in_tensor in range(0, TENSOR_K, TILE_K):
                k_size_tile = min(TILE_K, TENSOR_K - tile_k_start_in_tensor)
                k_size_tile_align_k0 = (k_size_tile + K0 - 1) // K0 * K0 # K0 对齐
                
                # DDR -> L1 排布变换 (MK -> K1MK0)
                left_k1mk0_tile = MK2K1MK0_DDR2L1(left, TENSOR_M, TENSOR_K, tile_m_start_in_tensor, tile_k_start_in_tensor, m_size_tile, k_size_tile)
                right_k1nk0_tile = MK2K1MK0_DDR2L1(right_nk, TENSOR_N, TENSOR_K, tile_n_start_in_tensor, tile_k_start_in_tensor, n_size_tile, k_size_tile)
                
                # 以下是 L1 -> L0 的 Slice 级切分
                for slice_n_start_in_tile in range(0, n_size_tile, SLICE_N):
                    n_size_slice = min(SLICE_N, n_size_tile - slice_n_start_in_tile)
                    bias_n1n0_pmb = N2N1N0_L12PMB(bias_n_tile, slice_n_start_in_tile, n_size_slice)
                    deq_n1n0_pmb = N2N1N0_L12PMB(deq_n_tile, slice_n_start_in_tile, n_size_slice)
                    
                    for slice_m_start_in_tile in range(0, m_size_tile, SLICE_M):
                        m_size_slice = min(SLICE_M, m_size_tile - slice_m_start_in_tile)
                        
                        for slice_k_start_in_tile in range(0, k_size_tile_align_k0, SLICE_K):
                            k_size_slice = min(SLICE_K, k_size_tile_align_k0 - slice_k_start_in_tile)
                            assert(k_size_slice % K0 == 0)
                            k1_size_slice = k_size_slice // K0
                            slice_k1_start_in_tile = slice_k_start_in_tile // K0
                            
                            # 从 L1 取出数据到 L0 Buffer (LMB/RMB)
                            left_m1k1m0k0_lmb = K1MK02M1K1M0K0_L12LMB(left_k1mk0_tile, m_size_tile, slice_m_start_in_tile, slice_k1_start_in_tile, m_size_slice, k1_size_slice)
                            right_n1k1n0k0_rmb = K1NK02N1K1N0K0_L12RMB(right_k1nk0_tile, n_size_tile, slice_n_start_in_tile, slice_k1_start_in_tile, n_size_slice, k1_size_slice)
                            
                            # 控制信号生成
                            # 1. 只有在当前 Tile 第一次计算时加偏置
                            bias_en = (tile_k_start_in_tensor == 0 and slice_k_start_in_tile == 0)
                            # 2. 除了第一次计算，其他时候都需要累加之前的部分和
                            psum_en = (tile_k_start_in_tensor != 0 or slice_k_start_in_tile != 0)
                            # 3. 只有当整个矩阵的 K 维度都被处理完后，才做反量化 (或者在当前架构中，是每个 Tile 计算完 K 后)
                            is_last_k_in_tile = (slice_k_start_in_tile + k_size_slice >= k_size_tile_align_k0)
                            is_last_tile_in_k = (tile_k_start_in_tensor + k_size_tile >= TENSOR_K)
                            deq_en = is_last_k_in_tile and is_last_tile_in_k
                            
                            psb_addr = (slice_m_start_in_tile//SLICE_M * ceil_div(n_size_tile, SLICE_N) + slice_n_start_in_tile//SLICE_N)
                            # 调用计算核心
                            result_m2n2m1n1m0n0_psb[psb_addr] = matmul_m1k1m0k0_n1k1n0k0(\
                            result_m2n2m1n1m0n0_psb[psb_addr], left_m1k1m0k0_lmb, right_n1k1n0k0_rmb, bias_n1n0_pmb, deq_n1n0_pmb, left_m1k1m0k0_lmb.shape[0], right_n1k1n0k0_rmb.shape[0], left_m1k1m0k0_lmb.shape[1], bias_en, psum_en, deq_en)
                            
                            # 写回 DDR
                            if(deq_en):
                                result_mn = M1N1M0N02MN_PSB2DDR(result_mn, result_m2n2m1n1m0n0_psb[psb_addr], TENSOR_N, tile_m_start_in_tensor+slice_m_start_in_tile, tile_n_start_in_tensor+slice_n_start_in_tile, m_size_slice, n_size_slice)
    
    # 验证结果
    golden_mn = matmul_mk_kn(left, right, bias, deq)
    compare(result_mn, golden_mn)
    return True

for i in range(10):
    test_matmul()