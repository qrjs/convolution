from lib import *

# 将 (Channel, Height, Width) 格式的输入特征图 (IFM) 转换为矩阵 (M, K)
# 这就是著名的 im2col (Image to Column) 变换
def CHW2MK(ifm, layer):
    out_height = layer['out_height']
    out_width = layer['out_width']
    in_height = layer['in_height']
    in_width = layer['in_width']
    in_channel = layer['in_channel']
    kernel_h = layer['kernel_h']
    kernel_w = layer['kernel_w']
    stride_h = layer['stride_h']
    stride_w = layer['stride_w']
    pad_h = layer['pad_h']
    pad_w = layer['pad_w']
    dilation_h = layer['dilation_h']
    dilation_w = layer['dilation_w']
    
    # M: 卷积滑动的次数 (输出特征图的像素点总数)
    M = out_height * out_width
    # K: 每一个卷积核窗口内的元素数量 (C * Kh * Kw)
    K = in_channel * kernel_h * kernel_w

    # 初始化矩阵
    ofm = np.zeros((M, K))
    
    # 遍历输出特征图的每一个坐标 (oh, ow) -> 对应矩阵的一行
    for oh in range(out_height):
        for ow in range(out_width):
            m_idx = oh * out_width + ow
            # 遍历卷积核的每一个坐标 (kh, kw) -> 对应矩阵的一列中的一部分
            for kh in range(kernel_h):
                for kw in range(kernel_w):
                    # 计算当前卷积核坐标对应输入图上的坐标 (考虑 stride 和 dilation)
                    ih = oh * stride_h - pad_h + kh * dilation_h
                    iw = ow * stride_w - pad_w + kw * dilation_w
                    
                    # 如果 ih, iw 在输入图范围内且不是 pad 部分
                    if (0 <= ih < in_height) and (0 <= iw < in_width):
                        # 遍历输入通道
                        for ic in range(in_channel):
                            # 计算 K 维度的索引： ic * Kh * Kw + kh * Kw + kw
                            k_idx = ic * kernel_h * kernel_w + kh * kernel_w + kw
                            ofm[m_idx][k_idx] = ifm[ic][ih][iw]
    return ofm

def test_im2col():
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
    if(out_height <= 0 or out_width <= 0):
        return False
    # 定义卷积层参数
    layer = {
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
    'dilation_w': dilation_w
    }
    print(layer)

    # 创建输入数据
    ifm = np.arange(in_channel * in_height * in_width).reshape(in_channel, in_height, in_width) + 1
    weight = np.arange(out_channel * in_channel * kernel_h * kernel_w).reshape(out_channel, in_channel, kernel_h, kernel_w) + 1
    bias = np.arange(out_channel) + 1
    ifm_mk = CHW2MK(ifm, layer)
    weight_nk = weight.reshape((out_channel, in_channel*kernel_h*kernel_w))
    ofm_mn = np.matmul(ifm_mk, weight_nk.transpose())
    ofm_nm = ofm_mn.transpose()
    for i in range(out_channel):
        ofm_nm[i] += bias[i]
    ofm_golden = golden_convolution(ifm, weight, bias, layer)
    compare(ofm_nm.flatten(), ofm_golden.flatten())
    return ofm_nm

for i in range(10):
    test_im2col()
