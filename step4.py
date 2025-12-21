from lib import *

def BHWC2C1BHWC0_DDR2L1(ifm, batch, tensor_h, tensor_w, tensor_c, start_tensor_h, start_tensor_w, start_tensor_c, tile_h, tile_w, tile_c):
    C1 = ceil_div(tile_c, C0)
    ifm_C1BHWC0 = np.zeros((C1, batch, tile_h, tile_w, C0))
    assert(ifm.shape[1] == tensor_h)
    assert(ifm.shape[2] == tensor_w)
    assert(ifm.shape[3] == tensor_c)
    for c1 in range(C1):
        for b in range(batch):
            for h in range(tile_h):
                for w in range(tile_w):
                    for c0 in range(C0):

    return ifm_C1BHWC0.reshape((C1, batch, tile_h, tile_w, C0))

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

def C1BHWC02M1K1M0K0_L12L0(ifm, batch, slice_oh, slice_ow, slice_c1, start_tile_h, start_tile_w, start_tile_c1, kernel_h, kernel_w, stride_h, stride_w, pad_t, pad_b, pad_l, pad_r, dilation_h, dilation_w, padding_value=0):
    slice_ih = (slice_oh - 1) * stride_h + dilation_h * (kernel_h - 1) - pad_t - pad_b + 1
    slice_iw = (slice_ow - 1) * stride_w + dilation_w * (kernel_w - 1) - pad_l - pad_r + 1
    C1 = slice_c1
    M = batch * slice_oh * slice_ow
    M1 = ceil_div(M, M0)
    K1 = C1*kernel_h*kernel_w
    ifm_m1k1m0k0 = np.zeros((M1, K1, M0, K0))
    for ic1 in range(C1):
        for kh in range(kernel_h):
            for kw in range(kernel_w):
                for b in range(batch):
                    for oh in range(slice_oh):
                        for ow in range(slice_ow):
                            ih = 
                            iw = 
                            m_index = 
                            k1_idx = 
                            m1_idx = m_index // M0
                            m0_idx = m_index % M0
                            if (0 <= ih < slice_ih) and (0 <= iw < slice_iw):
                                
                            else:
                                for k0_idx in range(K0):
                                    ifm_m1k1m0k0[m1_idx][k1_idx][m0_idx][k0_idx] = padding_value
    return ifm_m1k1m0k0

def test_convolution():
    batch = np.random.randint(1, 4)
    # 对应OpenNPU的切分策略中的H_L2, W_L2, IC_L2, OC_L2
    in_height = np.random.randint(3, 100)
    in_width = np.random.randint(3, 100)
    in_channel = np.random.randint(3, 100)
    out_channel = np.random.randint(3, 100)
    kernel_h = np.random.randint(1,7)
    kernel_w = np.random.randint(1,7)
    stride_h = np.random.randint(1,7)
    stride_w = np.random.randint(1,7)
    pad_h = np.random.randint(0, kernel_h)
    pad_w = np.random.randint(0, kernel_w)
    dilation_h = np.random.randint(1,7)
    dilation_w = np.random.randint(1,7)
    out_height = (in_height + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) // stride_h + 1
    out_width = (in_width + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) // stride_w + 1
    if(out_height <= 2 or out_width <= 2):
        return False
    H_L1 = np.random.randint(2, out_height)
    W_L1 = np.random.randint(2, out_width)
    CI_L1 = np.random.randint(2, in_channel)
    CO_L1 = ceil_align(np.random.randint(1, out_channel), N0)
    H_L0 = np.random.randint(1, H_L1)
    W_L0 = np.random.randint(1, W_L1)
    CI1_L0 = ceil_div(np.random.randint(1, CI_L1), K0)
    CO_L0 = ceil_align(np.random.randint(1, CO_L1), N0)
    # 定义卷积层参数
    layer = {
    'batch': batch,
    'in_height': in_height,
    'in_width': in_width,
    'out_height': out_height,
    'out_width': out_width,
    'in_channel': in_channel,
    'out_channel': out_channel,
    'kernel_h': kernel_h,
    'kernel_w': kernel_w,
    'stride_h': stride_h,
    'stride_w': stride_w,
    'pad_h': pad_h,
    'pad_w': pad_w,
    'dilation_h': dilation_h,
    'dilation_w': dilation_w,
    'CO_L1': CO_L1,
    'H_L1': H_L1,
    'W_L1': W_L1,
    'CI_L1': CI_L1,
    'CO_L0': CO_L0,
    'H_L0': H_L0,
    'W_L0': W_L0,
    'CI1_L0': CI1_L0
    }
    print(layer)

    # 创建输入数据
    ifm_BCHW = np.random.randn(batch, in_channel, in_height, in_width)
    weight = np.random.randn(out_channel, in_channel, kernel_h, kernel_w)
    bias = np.random.randn(out_channel)
    deq = np.random.rand(out_channel)
    ofm_BHWC = np.zeros((batch, out_height, out_width, out_channel))
    # 先转成C1HWC0, 再使用im2col和gemm
    ifm_BHWC = ifm_BCHW.transpose((0, 2, 3, 1))
    for tile_oc_start_in_tensor in range(0, out_channel, CO_L1):
        for tile_oh_start_in_tensor in range(0, out_height, H_L1):
            for tile_ow_start_in_tensor in range(0, out_width, W_L1):
                # 分块操作，以0C_L1,H_L1,W_L1为长宽高为大小的矩形进行分块
                oc_size_tile = min(CO_L1, out_channel - tile_oc_start_in_tensor)
                oh_size_tile = min(H_L1, out_height - tile_oh_start_in_tensor)
                ow_size_tile = min(W_L1, out_width - tile_ow_start_in_tensor)
                # im2col操作后特征图使用矩阵表达的维度: M=Ho*Wo, K=CiKhKw, N=Co
                # 倒推做im2col膨胀前使用矩阵表达的维度: M=Hi*Wi, K=Ci, N=Co
                # 矩阵在L1的layout是K1MK0, 特征图在L1的layout是Ci1HiWiCi0
                M = batch * H_L0 * W_L0
                N = oc_size_tile
                # M0，N0，K0是Cube Unit进行一次矩阵乘法最小的粒度。M1、N1、K1是矩阵分块的索引，比如M1=0, N1=0指输出结果矩阵的第一个分块。
                # 向上取整显示出了矩阵的对齐策略: 如果M和N不是M0和N0的整数倍，需要填充到M0和N0的整数倍。
                M1 = ceil_div(M, M0)
                N1 = ceil_div(N, N0)
                # psb: partial sum buffer，卷积产生的中间数据(CU产生的部分和存入PSB，CU计算时从PSB加载部分和，VU从PSB读数据写入GM)
                result_m2n2m1n1m0n0_psb = np.zeros((batch*ceil_div(oh_size_tile, H_L0)*ceil_div(ow_size_tile, W_L0)*ceil_div(oc_size_tile, CO_L0), M1, N1, M0, N0))
                for tile_ic_start_in_tensor in range(0, in_channel, CI_L1):
                    # tile_oh_end_tensor指本次L1计算的特征图结束行在L2中的坐标
                    tile_oh_end_in_tensor = min(tile_oh_start_in_tensor + H_L1, out_height) - 1
                    tile_ow_end_in_tensor = min(tile_ow_start_in_tensor + W_L1, out_width) - 1
                    # 根据卷积操作输出特征图的起始坐标和终止坐标倒推输入特征图的起始坐标和终止坐标
                    tile_ih_start_in_tensor = max(tile_oh_start_in_tensor * stride_h - pad_h, 0)
                    tile_iw_start_in_tensor = max(tile_ow_start_in_tensor * stride_w - pad_w, 0)
                    tile_ih_end_in_tensor = min(tile_oh_end_in_tensor * stride_h - pad_h + dilation_h * (kernel_h - 1), in_height - 1)
                    tile_iw_end_in_tensor = min(tile_ow_end_in_tensor * stride_w - pad_w + dilation_w * (kernel_w - 1), in_width - 1)
                    tile_ic_end_in_tensor = min(tile_ic_start_in_tensor + CI_L1, in_channel) - 1
                    # 计算本次L1计算的输入特征图的大小
                    ih_size_tile = max(tile_ih_end_in_tensor - tile_ih_start_in_tensor + 1, 0)
                    iw_size_tile = max(tile_iw_end_in_tensor - tile_iw_start_in_tensor + 1, 0)
                    ic_size_tile = max(tile_ic_end_in_tensor - tile_ic_start_in_tensor + 1, 0)
                    # ci1_size_tile是本次L1计算的输入特征图从DDR搬运到L1变成C1BHWC0后的C1的大小
                    ci1_size_tile = ceil_div(ic_size_tile, K0)
                    ifm_C1BHWC0_tile = BHWC2C1BHWC0_DDR2L1(ifm_BHWC, batch, in_height, in_width, in_channel, tile_ih_start_in_tensor, tile_iw_start_in_tensor, tile_ic_start_in_tensor, ih_size_tile, iw_size_tile, ic_size_tile)
                    weight_k1nk0_tile = OCICKhKw2IC1KhKwOIC0(weight, out_channel, in_channel, kernel_h, kernel_w, tile_oc_start_in_tensor, tile_ic_start_in_tensor, oc_size_tile, ic_size_tile)
                    for slice_oc_start_in_tile in range(0, oc_size_tile, CO_L0):
                        for slice_oh_start_in_tile in range(0, oh_size_tile, H_L0):
                            for slice_ow_start_in_tile in range(0, ow_size_tile, W_L0):
                                for slice_ci1_start_in_tile in range(0, ci1_size_tile, CI1_L0):
                                    slice_oc_start_in_tensor = tile_oc_start_in_tensor + slice_oc_start_in_tile
                                    slice_oh_start_in_tensor = tile_oh_start_in_tensor + slice_oh_start_in_tile
                                    slice_ow_start_in_tensor = tile_ow_start_in_tensor + slice_ow_start_in_tile
                                    slice_oh_end_in_tensor = min(slice_oh_start_in_tensor + H_L0, tile_oh_start_in_tensor + oh_size_tile) - 1
                                    slice_ow_end_in_tensor = min(slice_ow_start_in_tensor + W_L0, tile_ow_start_in_tensor + ow_size_tile) - 1
                                    slice_ih_start_in_tensor = max(slice_oh_start_in_tensor * stride_h - pad_h, 0)
                                    slice_iw_start_in_tensor = max(slice_ow_start_in_tensor * stride_w - pad_w, 0)
                                    slice_ih_start_in_tile = slice_ih_start_in_tensor - tile_ih_start_in_tensor
                                    slice_iw_start_in_tile = slice_iw_start_in_tensor - tile_iw_start_in_tensor
                                    slice_ow_size = slice_ow_end_in_tensor - slice_ow_start_in_tensor + 1
                                    slice_oh_size = slice_oh_end_in_tensor - slice_oh_start_in_tensor + 1
                                    slice_ci1_size = min(ci1_size_tile - slice_ci1_start_in_tile, CI1_L0)
                                    slice_oc_size = min(oc_size_tile - slice_oc_start_in_tile, CO_L0)
                                    # pad_t, pad_b, pad_l, pad_r分别表示上下左右填充的行数
                                    if(slice_oh_start_in_tensor * stride_h - pad_h < 0):
                                        slice_pad_t = 
                                    else:
                                        slice_pad_t = 0
                                    if():
                                        slice_pad_b = 
                                    else:
                                        slice_pad_b = 0
                                    if(slice_ow_start_in_tensor * stride_w - pad_w < 0):
                                        slice_pad_l = 
                                    else:
                                        slice_pad_l = 0
                                    if():
                                        slice_pad_r = 
                                    else:
                                        slice_pad_r = 0
                                    # IC方向的第一次计算需要加偏置
                                    bias_en = (tile_ic_start_in_tensor==0) and (slice_ci1_start_in_tile==0)
                                    # 思考一下什么情况需要跟上一次的结果累加
                                    psum_en = 
                                    # 卷积计算不在KhKw维度切分，只有在卷积计算的IC方向彻底做完，对应矩阵的K维度彻底做完才能进行dequant
                                    deq_en = 
                                    ifm_M1K1M0K0 = C1BHWC02M1K1M0K0_L12L0(ifm_C1BHWC0_tile, batch, slice_oh_size, slice_ow_size, slice_ci1_size, slice_ih_start_in_tile, slice_iw_start_in_tile, slice_ci1_start_in_tile, kernel_h, kernel_w, stride_h, stride_w, slice_pad_t, slice_pad_b, slice_pad_l, slice_pad_r, dilation_h, dilation_w)
                                    weight_N1K1N0K0_slice = K1NK02N1K1N0K0_L12RMB(weight_k1nk0_tile, oc_size_tile, slice_oc_start_in_tile, slice_ci1_start_in_tile*kernel_h*kernel_w, slice_oc_size, slice_ci1_size*kernel_h*kernel_w)
                                    bias_N1N0 = N2N1N0_L12PMB(bias, slice_oc_start_in_tensor, slice_oc_size)
                                    deq_N1N0 = N2N1N0_L12PMB(deq, slice_oc_start_in_tensor, slice_oc_size)
                                    psb_addr = 
                                    result_m2n2m1n1m0n0_psb[psb_addr] = matmul_m1k1m0k0_n1k1n0k0(\
                                    result_m2n2m1n1m0n0_psb[psb_addr], ifm_M1K1M0K0, weight_N1K1N0K0_slice, bias_N1N0, deq_N1N0, ifm_M1K1M0K0.shape[0], weight_N1K1N0K0_slice.shape[0], ifm_M1K1M0K0.shape[1], bias_en, psum_en, deq_en)
                                    if(deq_en):
                                        ofm_BHWC = M1N1M0N02BHWC_PSB2DDR(ofm_BHWC, result_m2n2m1n1m0n0_psb[psb_addr], batch, out_height, out_width, out_channel, slice_oh_start_in_tensor, slice_ow_start_in_tensor, slice_oc_start_in_tensor, slice_oh_size, slice_ow_size, slice_oc_size)
    # 使用torch验证
    ifm_torch = torch.tensor(ifm_BCHW, dtype=torch.float32)
    weight_torch = torch.tensor(weight, dtype=torch.float32)
    bias_torch = torch.tensor(bias, dtype=torch.float32)
    deq_torch = torch.tensor(deq, dtype=torch.float32)
    ofm_torch = nn.functional.conv2d(ifm_torch, weight_torch, bias_torch, stride=(stride_h, stride_w), padding=(pad_h, pad_w), dilation=(dilation_h, dilation_w))
    ofm_torch = ofm_torch * deq_torch.view(1, -1, 1, 1)
    ofm_BCHW = ofm_BHWC.reshape((batch, out_height, out_width, out_channel)).transpose((0, 3, 1, 2))
    test_pass = compare(ofm_BCHW.flatten(), ofm_torch.numpy().flatten())
    return test_pass

for i in range(20):
    test_convolution()