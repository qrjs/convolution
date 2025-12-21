from lib import *

def MK2K1MK0_DDR2L1(matrix_mk, tensor_m, tensor_k, start_m, start_k, tile_m, tile_k):
    K1 = ceil_div(tile_k, K0)
    matrix_k1mk0 = np.zeros((K1, tile_m, K0))
    for k1 in range(K1):
        for m in range(tile_m):
            for k0 in range(K0):

    return matrix_k1mk0

def matmul_m1k1m0k0_n1k1n0k0(m1n1m0n0, m1k1m0k0, n1k1n0k0, bias_n1n0, deq_n1n0, M1, N1, K1, bias_en, psum_en, deq_en):
    assert(m1k1m0k0.shape[1] == n1k1n0k0.shape[1])
    assert(m1k1m0k0.shape[3] == n1k1n0k0.shape[3])
    for m1 in range(M1):
        for n1 in range(N1):
            temp = np.zeros((M0, N0))
            for k1 in range(K1):

            if(bias_en):
                for n0 in range(N0):

            if(psum_en):
                m1n1m0n0[m1][n1] += temp
            else:
                m1n1m0n0[m1][n1] = temp
    if(deq_en):

    return m1n1m0n0 

def test_matmul():
    # 定义输入参数
    TENSOR_M = np.random.randint(3, 100)
    TENSOR_N = np.random.randint(3, 100)
    TENSOR_K = np.random.randint(3, 100)
    TILE_M = np.random.randint(2, TENSOR_M)
    TILE_N = np.random.randint(2, TENSOR_N)
    TILE_K = np.random.randint(2, TENSOR_K)

    SLICE_M = ceil_align(np.random.randint(1, TILE_M), M0)
    SLICE_N = ceil_align(np.random.randint(1, TILE_M), N0)
    SLICE_K = ceil_align(np.random.randint(1, TILE_M), K0)
    SLICE_M1 = ceil_div(SLICE_M, M0)
    SLICE_N1 = ceil_div(SLICE_N, N0)

    # 定义矩阵乘法参数
    layer = {
        'M': TENSOR_M,
        'N': TENSOR_N,
        'K': TENSOR_K
    }
    print(layer)

    # 创建输入数据
    left = np.random.randint(-128, 127, size=(TENSOR_M, TENSOR_K))
    right = np.random.randint(-128, 127, size=(TENSOR_K, TENSOR_N))
    bias = np.random.randint(-128, 127, size=(TENSOR_N))
    deq = np.random.rand(TENSOR_N)
    right_nk = right.transpose()
    result_mn = np.zeros((TENSOR_M, TENSOR_N))

    for tile_n_start_in_tensor in range(0, TENSOR_N, TILE_N):
        n_size_tile = min(TILE_N, TENSOR_N - tile_n_start_in_tensor)
        bias_n_tile = bias[tile_n_start_in_tensor:tile_n_start_in_tensor+n_size_tile]
        deq_n_tile = deq[tile_n_start_in_tensor:tile_n_start_in_tensor+n_size_tile]
        for tile_m_start_in_tensor in range(0, TENSOR_M, TILE_M):
            m_size_tile = min(TILE_M, TENSOR_M - tile_m_start_in_tensor)
            result_m2n2m1n1m0n0_psb = np.zeros((ceil_div(m_size_tile, SLICE_M)*ceil_div(n_size_tile, SLICE_N), SLICE_M1, SLICE_N1, M0, N0))
            for tile_k_start_in_tensor in range(0, TENSOR_K, TILE_K):
                k_size_tile = min(TILE_K, TENSOR_K - tile_k_start_in_tensor)
                k_size_tile_align_k0 = (k_size_tile + K0 - 1) // K0 * K0
                left_k1mk0_tile = MK2K1MK0_DDR2L1(left, TENSOR_M, TENSOR_K, tile_m_start_in_tensor, tile_k_start_in_tensor, m_size_tile, k_size_tile)
                right_k1nk0_tile = MK2K1MK0_DDR2L1(right_nk, TENSOR_N, TENSOR_K, tile_n_start_in_tensor, tile_k_start_in_tensor, n_size_tile, k_size_tile)
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
                            left_m1k1m0k0_lmb = K1MK02M1K1M0K0_L12LMB(left_k1mk0_tile, m_size_tile, slice_m_start_in_tile, slice_k1_start_in_tile, m_size_slice, k1_size_slice)
                            right_n1k1n0k0_rmb = K1NK02N1K1N0K0_L12RMB(right_k1nk0_tile, n_size_tile, slice_n_start_in_tile, slice_k1_start_in_tile, n_size_slice, k1_size_slice)
                            bias_en = 
                            psum_en = 
                            deq_en = 
                            psb_addr = (slice_m_start_in_tile//SLICE_M * ceil_div(n_size_tile, SLICE_N) + slice_n_start_in_tile//SLICE_N)
                            result_m2n2m1n1m0n0_psb[psb_addr] = matmul_m1k1m0k0_n1k1n0k0(\
                            result_m2n2m1n1m0n0_psb[psb_addr], left_m1k1m0k0_lmb, right_n1k1n0k0_rmb, bias_n1n0_pmb, deq_n1n0_pmb, left_m1k1m0k0_lmb.shape[0], right_n1k1n0k0_rmb.shape[0], left_m1k1m0k0_lmb.shape[1], bias_en, psum_en, deq_en)
                            if(deq_en):
                                result_mn = M1N1M0N02MN_PSB2DDR(result_mn, result_m2n2m1n1m0n0_psb[psb_addr], TENSOR_N, tile_m_start_in_tensor+slice_m_start_in_tile, tile_n_start_in_tensor+slice_n_start_in_tile, m_size_slice, n_size_slice)
    golden_mn = matmul_mk_kn(left, right, bias, deq)
    compare(result_mn, golden_mn)
    return True

for i in range(10):
    test_matmul()