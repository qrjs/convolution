import numpy as np

def ceil_div(x, y):
    return (x + y - 1) // y

def test_matmul():
    # 随机生成 M, K, N 的维度
    M = np.random.randint(3, 100)
    K = np.random.randint(3, 100)
    N = np.random.randint(3, 100)
    
    # 定义 L0 层的 buffer 大小 (分块大小)
    M0 = 3
    N0 = 4
    K0 = 5    
    
    # 计算 L1 层的维度 (分块数量)
    M1 = ceil_div(M, M0)
    N1 = ceil_div(N, N0)
    K1 = ceil_div(K, K0)
    
    # 定义矩阵乘法参数
    layer = {
        'M': M,
        'N': N,
        'K': K,
        'M1': M1,
        'N1': N1,
        'K1': K1
    }
    print(layer)

    # 初始化随机输入矩阵 (DDR层数据，Layout: MK, NK)
    left_mk = np.random.randint(-128, 127, size=(M, K))
    right_nk = np.random.randint(-128, 127, size=(N, K))
    
    # 结果矩阵 (L1层结果，Layout: M1N1M0N0)
    result_m1n1m0n0 = np.zeros((M1, N1, M0, N0))

    # --- 数据重排 (Data Packing / Layout Transformation) ---
    # 将输入矩阵从 DDR 的线性排布转换为 TPU 内部友好的分块排布
    
    # Left Matrix (A): MK -> M1K1M0K0
    # 1. Padding: 填充到 M1*M0, K1*K0
    left_m1m0k1k0 = np.zeros((M1*M0, K1*K0))
    left_m1m0k1k0[:M, :K] = left_mk
    # 2. Reshape & Transpose: (M1*M0, K1*K0) -> (M1, M0, K1, K0) -> (M1, K1, M0, K0)
    left_m1k1m0k0 = left_m1m0k1k0.reshape(M1, M0, K1, K0).transpose(0, 2, 1, 3)

    # Right Matrix (B): NK -> N1K1N0K0
    # 1. Padding: 填充到 N1*N0, K1*K0
    right_n1n0k1k0 = np.zeros((N1*N0, K1*K0))
    right_n1n0k1k0[:N, :K] = right_nk
    # 2. Reshape & Transpose: (N1*N0, K1*K0) -> (N1, N0, K1, K0) -> (N1, K1, N0, K0)
    right_n1k1n0k0 = right_n1n0k1k0.reshape(N1, N0, K1, K0).transpose(0, 2, 1, 3)

    # --- 分块矩阵乘法 (Tiled Matrix Multiplication) ---
    # 模拟硬件的循环执行顺序
    for m1 in range(M1):
        for n1 in range(N1):
            # 初始化累加器 (Accumulator)
            temp = np.zeros((M0, N0)) 
            for k1 in range(K1):
                # 核心计算单元 (L0层计算): M0xK0 @ K0xN0 -> M0xN0
                # left_m1k1m0k0[m1][k1] shape: (M0, K0)
                # right_n1k1n0k0[n1][k1] shape: (N0, K0) -> Transpose to (K0, N0)
                temp += left_m1k1m0k0[m1][k1] @ right_n1k1n0k0[n1][k1].T
            # 将累加结果存入 L1 结果 buffer
            result_m1n1m0n0[m1][n1] = temp

    # --- 结果重构 (Result Reconstruction) ---
    # 将分块结果转换回线性排布 (M1N1M0N0 -> MN)
    result_mn = np.zeros((M, N))
    for m1 in range(M1):
        for n1 in range(N1):
            for m0 in range(M0):
                for n0 in range(N0):
                    # 计算全局坐标
                    m = m1 * M0 + m0
                    n = n1 * N0 + n0
                    # 去除 padding 部分
                    if m < M and n < N:
                        result_mn[m][n] = result_m1n1m0n0[m1][n1][m0][n0]
    
    # --- 验证 (Verification) ---
    # 与标准矩阵乘法结果对比
    # 注意：right_nk 在生成时是 (N, K)，标准乘法需要 B (K, N)，所以这里用 right_nk.transpose() 即 (K, N)
    # C(M, N) = A(M, K) * B(K, N)
    result_golden = np.matmul(left_mk, right_nk.transpose())
    
    diff = np.abs(result_golden - result_mn)
    if(diff.sum() == 0):
        print('Pass')
    else:
        print('Fail')
        print('Golden:', result_golden.shape)
        print('Result:', result_mn.shape)

for i in range(10):
    test_matmul()