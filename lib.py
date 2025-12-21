import numpy as np
import torch
import torch.nn as nn
M0 = 3
N0 = 4
K0 = C0 = 5

def ceil_div(x, y):
    return (x + y - 1) // y

def ceil_align(x, y):
    return ceil_div(x, y) * y

def convolution(ifm, weight, bias, layer):
    out_height = layer['out_height']
    out_width = layer['out_width']
    in_height = layer['in_height']
    in_width = layer['in_width']
    in_channel = layer['in_channel']
    out_channel = layer['out_channel']
    kernel_h = layer['kernel_h']
    kernel_w = layer['kernel_w']
    stride_h = layer['stride_h']
    stride_w = layer['stride_w']
    pad_h = layer['pad_h']
    pad_w = layer['pad_w']
    dilation_h = layer['dilation_h']
    dilation_w = layer['dilation_w']
    ofm = np.zeros((out_channel, out_height, out_width))
    for oh in range(out_height):
        for ow in range(out_width):
            for oc in range(out_channel):
                y_data = 0
                for kh in range(kernel_h):
                    for kw in range(kernel_w):
                        ih = oh * stride_h - pad_h + kh * dilation_h
                        iw = ow * stride_w - pad_w + kw * dilation_w
                        if (0 <= ih < in_height) and (0 <= iw < in_width):
                            for ic in range(in_channel):
                                ifm_index = ic * in_height * in_width + ih * in_width + iw
                                wgt_index = oc * kernel_h * kernel_w * in_channel + ic * kernel_h * kernel_w + kh * kernel_w + kw
                                x_data = ifm[ifm_index]
                                w_data = weight[wgt_index]
                                y_data += x_data * w_data
                ofm[oc][oh][ow] = y_data + bias[oc]
    return ofm

def C1BHWC02M1K1M0K0(ifm, layer):
    batch = layer['batch']
    in_channel = layer['in_channel']
    in_height = layer['in_height']
    in_width = layer['in_width']
    out_height = layer['out_height']
    out_width = layer['out_width']
    kernel_h = layer['kernel_h']
    kernel_w = layer['kernel_w']
    stride_h = layer['stride_h']
    stride_w = layer['stride_w']
    pad_h = layer['pad_h']
    pad_w = layer['pad_w']
    dilation_h = layer['dilation_h']
    dilation_w = layer['dilation_w']
    C1 = ceil_div(in_channel, C0)
    M = batch * out_height * out_width
    M1 = ceil_div(M, M0)
    K1 = C1*kernel_h*kernel_w
    ifm_m1k1m0k0 = np.zeros((M1, K1, M0, K0))
    for ic1 in range(C1):
        for kh in range(kernel_h):
            for kw in range(kernel_w):
                for b in range(batch):
                    for oh in range(out_height):
                        for ow in range(out_width):
                            ih = kh * dilation_h - pad_h + oh * stride_h
                            iw = kw * dilation_w - pad_w + ow * stride_w
                            if (0 <= ih < in_height) and (0 <= iw < in_width):
                                m_index = b*out_height*out_width + oh*out_width + ow
                                k1_idx = ic1 * kernel_h * kernel_w + kh * kernel_w + kw
                                m1_idx = m_index // M0
                                m0_idx = m_index % M0
                                ifm_m1k1m0k0[m1_idx][k1_idx][m0_idx] = ifm[ic1][b][ih][iw]
    return ifm_m1k1m0k0

# NK -> K1NK0
def OCICKhKw2IC1KhKwOIC0(weight, out_channel, in_channel, kernel_h, kernel_w, oc_start, ic_start, oc_size, ic_size):
    assert(weight.shape[0] == out_channel)
    assert(weight.shape[1] == in_channel)
    assert(weight.shape[2] == kernel_h)
    assert(weight.shape[3] == kernel_w)
    C1 = ceil_div(ic_size, C0)
    weight_OCICKhKw = weight[oc_start:oc_start + oc_size, ic_start:ic_start + ic_size]
    weight_OCKhKwIC = weight_OCICKhKw.transpose((0, 2, 3, 1))
    weight_OCKhKwIC_ = np.zeros((oc_size, kernel_h, kernel_w, C1*C0))
    for ic in range(ic_size):
        weight_OCKhKwIC_[:, :, :, ic] = weight_OCKhKwIC[:, :, :, ic]
    weight_OCKhKwCc = weight_OCKhKwIC_.reshape((oc_size, kernel_h, kernel_w, C1, C0))
    weight_IC1KhKwOIC0 = weight_OCKhKwCc.transpose((3, 1, 2, 0, 4)).reshape((C1*kernel_h*kernel_w, oc_size, C0))
    return weight_IC1KhKwOIC0

def golden_convolution(ifm_np, weight_np, bias_np, layer):
    pad_h = layer['pad_h']
    pad_w = layer['pad_w']
    stride_h = layer['stride_h']
    stride_w = layer['stride_w']
    dilation_h = layer['dilation_h']
    dilation_w = layer['dilation_w']
    # 为ifm增加一个batch维度
    ifm = torch.from_numpy(ifm_np).to(torch.float32)
    weight = torch.from_numpy(weight_np).to(torch.float32)
    bias = torch.from_numpy(bias_np).to(torch.float32)
    ofm = nn.functional.conv2d(ifm, weight, bias, stride=(stride_h, stride_w), padding=(pad_h, pad_w), dilation=(dilation_h, dilation_w))
    return ofm.numpy()

def compare(tensor_test, tensor_golden, threshold = 0.01):
    diff = np.abs(tensor_test.flatten() - tensor_golden.flatten())
    error_rate = diff / (np.abs(tensor_golden.flatten()) + 1e-10)
    if error_rate.mean() < threshold:
        print("pass")
        return True
    else:
        print("fail, error rate: ", np.max(error_rate))
        return False

def M1N1M0N02MN_PSB2DDR(matrix_mn, matrix_m1n1m0n0, tensor_n, start_tensor_m, start_tensor_n, slice_m, slice_n):
    M1 = ceil_div(slice_m, M0)
    N1 = ceil_div(slice_n, N0)
    assert(matrix_mn.shape[1] == tensor_n)
    for m1 in range(M1):
        for n1 in range(N1):
            for m0 in range(M0):
                for n0 in range(N0):
                    m = m1 * M0 + m0
                    n = n1 * N0 + n0
                    if(m < slice_m and n < slice_n):
                        matrix_mn[m+start_tensor_m][n+start_tensor_n] = matrix_m1n1m0n0[m1][n1][m0][n0]
    return matrix_mn

def K1MK02M1K1M0K0_L12LMB(matrix_k1mk0, tile_m, start_tile_m, start_tile_k1, slice_m, slice_k1):
    K1 = slice_k1
    M1 = ceil_div(slice_m, M0)
    assert(matrix_k1mk0.shape[1] == tile_m)
    matrix_m1k1m0k0 = np.zeros((M1, K1, M0, K0))
    for m1 in range(M1):
        for k1 in range(K1):
            for m0 in range(M0):
                m = m1 * M0 + m0
                if(m < slice_m):
                    matrix_m1k1m0k0[m1][k1][m0] = matrix_k1mk0[start_tile_k1+k1][start_tile_m+m]
    return matrix_m1k1m0k0

def K1NK02N1K1N0K0_L12RMB(matrix_k1nk0, tile_n, start_tile_n, start_tile_k1, slice_n, slice_k1):
    K1 = slice_k1
    N1 = ceil_div(slice_n, N0)
    assert(matrix_k1nk0.shape[1] == tile_n)
    matrix_n1k1n0k0 = np.zeros((N1, K1, N0, K0))
    for n1 in range(N1):
        for k1 in range(K1):
            for n0 in range(N0):
                n = n1 * N0 + n0
                if(n < slice_n):
                    matrix_n1k1n0k0[n1][k1][n0] = matrix_k1nk0[start_tile_k1+k1][start_tile_n+n]
    return matrix_n1k1n0k0

def N2N1N0_L12PMB(bias, start_n, tile_n):
    N1 = ceil_div(tile_n, N0)
    bias_N1N0 = np.zeros((N1, N0))
    for n1 in range(N1):
        for n0 in range(N0):
            n = n1 * N0 + n0
            if(n < tile_n):
                bias_N1N0[n1][n0] = bias[start_n+n]
    return bias_N1N0

def matmul_mk_kn(matrix_mk, matrix_kn, bias, deq):
    M = matrix_mk.shape[0]
    K = matrix_mk.shape[1]
    N = matrix_kn.shape[1]
    matrix_mn = np.matmul(matrix_mk, matrix_kn)
    matrix_mn = matrix_mn.astype(np.float32)
    for m in range(M):
        for n in range(N):
            matrix_mn[m][n] += bias[n]
            matrix_mn[m][n] *= deq[n]
    return matrix_mn

def M1N1M0N02BHWC_PSB2DDR(ofm_ddr, ofm_psb, batch, tensor_oh, tensor_ow, tensor_oc, start_tensor_oh, start_tensor_ow, start_tensor_oc, slice_oh, slice_ow, slice_oc):
    M = batch * slice_oh * slice_ow
    M1 = ceil_div(M, M0)
    N1 = ceil_div(slice_oc, N0)
    assert(ofm_ddr.shape[1] == tensor_oh)
    assert(ofm_ddr.shape[2] == tensor_ow)
    assert(ofm_ddr.shape[3] == tensor_oc)
    for m1 in range(M1):
        for n1 in range(N1):
            for m0 in range(M0):
                for n0 in range(N0):
                    m = m1 * M0 + m0
                    n = n1 * N0 + n0
                    b = m // (slice_oh * slice_ow)
                    oh = (m % (slice_oh * slice_ow)) // slice_ow
                    ow = (m % (slice_oh * slice_ow)) % slice_ow
                    if(b < batch and oh < slice_oh and ow < slice_ow and n < slice_oc):
                        ofm_ddr[b][start_tensor_oh + oh][start_tensor_ow + ow][start_tensor_oc + n] = ofm_psb[m1][n1][m0][n0]
    return ofm_ddr

def M1N1M0N02C1BHWC0_PSB2DDR(ofm_ddr, ofm_psb, batch, tensor_oh, tensor_ow, tensor_oc, start_tensor_oh, start_tensor_ow, start_tensor_oc, slice_oh, slice_ow, slice_oc):
    M = batch * slice_oh * slice_ow
    M1 = ceil_div(M, M0)
    N1 = ceil_div(slice_oc, N0)
    assert(ofm_ddr.shape[0] == ceil_div(tensor_oc, N0))
    assert(ofm_ddr.shape[2] == tensor_oh)
    assert(ofm_ddr.shape[3] == tensor_ow)
    for m1 in range(M1):
        for n1 in range(N1):
            for m0 in range(M0):
                m = m1 * M0 + m0
                b = m // (slice_oh * slice_ow)
                oh = (m % (slice_oh * slice_ow)) // slice_ow
                ow = (m % (slice_oh * slice_ow)) % slice_ow
                if(b < batch and oh < slice_oh and ow < slice_ow): 
                    ofm_ddr[n1 + start_tensor_oc//N0][b][start_tensor_oh + oh][start_tensor_ow + ow] = ofm_psb[m1][n1][m0]
    return ofm_ddr