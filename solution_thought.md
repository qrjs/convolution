# TPU 卷积运算模拟：从矩阵分块到完整卷积代码解析与通关指南

> [!NOTE]
> 本文档基于本项目 `step1.py` - `step4.py` 的**填空修改记录**，为你详细拆解整个项目的逻辑主线。在这个项目中，我们通过四个渐进式的 Python 脚本模拟了底层 AI 硬件（比如 NPU/TPU）处理矩阵乘法和卷积的过程。
> 
> 下面的讲解将结合你代码中填补的**关键公式与循环逻辑**，手把手带你理解为什么要这样算，以及怎样培养排错（Debug）的直觉。

---

## 📖 全局视角与硬件对齐理念

在看具体的长串多维循环前，先在脑海中建立这个物理模型：
1. **DDR（大仓库）**：数据以最线性的方式存放，比如普通的矩阵 `(M, K)` 或是 `(B, H, W, C)`。
2. **L1 Buffer（大货架）**：由于 DDR 读取太慢，我们要把数据搬运到 L1 中。为了方便之后极速地被计算核心吞吐，数据要被重组成一种包含**分块参数**的多维排布，例如 `(M1, K1, M0, K0)`。
3. **L0 Register / MAC 阵列（工作台）**：计算单元每次只能消化极小固定尺寸的矩阵块，本例中就是 `M0`、`N0`、`K0` 的大小（例如 `M0=3, N0=4, K0=5`）。

你的四个 `step` 脚本就是不断在实现：**如何正确切割 -> 搬运 -> 组合下标运算 -> 调用基础矩阵乘核心 -> 将结果写回**。

---

## 🛠️ Step 1: 矩阵分块与阵列核心计算模拟

**目标**：理解 `M1K1M0K0` 是什么，以及 L0 层面的微缩矩阵运算。

### 代码填空思路深度解析
在 `step1.py` 中，主要完成的是**核心循环**的逻辑和**坐标系拉平**的重构运算：

1. **底层小核心累加逻辑**：
   ```python
   # 针对当前 M1, N1 切片，初始化小方格累加器 temp
   temp = np.zeros((M0, N0)) 
   for k1 in range(K1):
       # 核心计算单元 (L0层计算): 用 left_m1k1m0k0 和 right 转置做矩阵乘
       temp += left_m1k1m0k0[m1][k1] @ right_n1k1n0k0[n1][k1].T
   ```
   **【思考】**：为什么这里内部还要有一个 `for k1`？因为我们算结果矩阵特定的一小块 `(M0, N0)`，等于输入矩阵提取特定的 `M0` 行乘权重特定的 `N0` 列；而这由于 K 维度太大，K 被切成了 `K1` 份，这就需要分别乘完这 `K1` 份并累积在 `temp` 中。

2. **全局结果逆向重组**：
   在把分块结果 `result_m1n1m0n0` 写回普通矩阵 `result_mn` 时，填写的核心公式如下：
   ```python
   # 计算出全局的绝对坐标
   m = m1 * M0 + m0
   n = n1 * N0 + n0
   # 由于我们切分时可能向外取整了（ceil_div），这里必须把补的 Padding 去掉：
   if m < M and n < N:
       result_mn[m][n] = result_m1n1m0n0[m1][n1][m0][n0]
   ```

---

## 🛠️ Step 2: 内存排布与控制信号（核心流控）

**目标**：理解从 `MK` 变成 `K1MK0` 的布局映射，以及 `bias`, `psum`, `deq` 触发时机。

### 代码填空思路深度解析
1. **DDR 到 L1 (将平铺二维变成多维排布)**：
   在 `MK2K1MK0_DDR2L1` 方法中填空的逻辑为：
   ```python
   # 获取原图 DDR 的一维坐标 K
   k = start_k + k1 * K0 + k0
   # 防越界保护
   if k < start_k + tile_k and k < tensor_k:
       matrix_k1mk0[k1][m][k0] = matrix_mk[start_m + m][k]
   ```
   **【思考】**：这是硬件中经典的 Layout Transformation 的软件模拟。`k1` 定位找哪个块，`k0` 定位找块内的哪个元素。

2. **控制信号的触发条件 (最容易写错的地方)**：
   这段填空直接决定了底层硬件数据流的生与死：
   - **`bias_en = (tile_k_start_in_tensor == 0 and slice_k_start_in_tile == 0)`**
     偏置加上去只能加一次。只有当这是该通道 `K` 维度的绝对第一次计算（即第一大块的第一小块），才启用 `bias_en`。
   - **`psum_en = (tile_k_start_in_tensor != 0 or slice_k_start_in_tile != 0)`**
     部分和使能信号。除了第一次计算不能累加上次结果，后面的每次切片循环，计算结果都要跟上次暂存格（PSB / Accumulator）的值进行累加！
   - **`deq_en = is_last_k_in_tile and is_last_tile_in_k`**
     反量化（dequantization）往往放在最后做。只有所有 `K` 维度都走完（意味着这个输出格子的加法都加透了），我们才允许进行后处理。

---

## 🛠️ Step 3: Im2Col（卷积转矩阵灵魂映射）

**目标**：理解如何不用 4 重循环滑动窗口写卷积，而是转为一次暴力的矩阵排布。

### 代码填空思路深度解析
在 `CHW2MK` 这个函数里，你需要把输入图片的 `(Channel, Height, Width)` 填成 `(M, K)` 形式。这里你填了两个极度关键的映射公式：
```python
# 核心填空 1：由输出坐标与核坐标，计算原图真实起点的 ih，iw
ih = oh * stride_h - pad_h + kh * dilation_h
iw = ow * stride_w - pad_w + kw * dilation_w

# 核心填空 2：边界防爆检查
if (0 <= ih < in_height) and (0 <= iw < in_width):
    ...
```
**【思考】**：
- `oh * stride_h` ：输出 `oh` 像素等于核在原图走过的宏观步长距离。
- `- pad_h`：因为左上角填充了补零层，真实的图片像素读取起始指针需要向“左上方”退回这么多值。
- `kh * dilation_h`：解决膨胀卷积时核内部的像素间距放大。
通过这个把图片“拉平”成一维宽长的举动，接下来的处理就全都可以套用前面的 `step1` 计算单元了！

---

## 🛠️ Step 4: 大整合与动态 Padding 边界计算 (Boss战)

**目标**：把 `im2col` 与多层分块嵌套。这里是报错的重灾区。

### 代码填空思路深度解析
我们来看看 `test_convolution()` 里面你填的四向 `padding` 切分逻辑：

1. **为什么切分出的大 Tile 或者 Slice 还要算 Padding?**
   原始参数里的 `pad_h` 和 `pad_w` 是针对**全图**的。但是图片被切成几十个方格小块运算时，只有**贴图边缘的方格**会取到补零部分，中间的方格是不需要补的！你需要通过比较绝对坐标，看它是不是出界了：
   
   **上边补零 (`slice_pad_t`) 填空思路**：
   ```python
   # 如果目前我这个切片的绝对起始点，向上走了 pad步后，变成负数了
   if(slice_oh_start_in_tensor * stride_h - pad_h < 0):
       # 收不回来那么多行，说明上面悬空，这些悬空的行数就成了补零数量
       slice_pad_t = pad_h - slice_oh_start_in_tensor * stride_h
   else:
       slice_pad_t = 0
   ```
   
   **右边补零 (`slice_pad_r`) 填空思路**：
   ```python
   # 这个切片的绝对终点 + 膨胀后整个核的真实横跨大小 >= 图片宽度全长
   if(slice_ow_end_in_tensor * stride_w - pad_w + dilation_w * (kernel_w - 1) >= in_width):
       # 说明超出右边界了，超出的部分就是我们这个块局部的 pad_r
       slice_pad_r = (slice_ow_end_in_tensor * stride_w - pad_w + dilation_w * (kernel_w - 1)) - (in_width - 1)
   else:
       slice_pad_r = 0
   ```

2. **复杂的一维内存切分编号 (`psb_addr`) 填空思路**：
   ```python
   psb_addr = (slice_oh_start_in_tile//H_L0 * ceil_div(ow_size_tile, W_L0) + slice_ow_start_in_tile//W_L0 + \
               slice_oc_start_in_tile//CO_L0 * ceil_div(oh_size_tile, H_L0) * ceil_div(ow_size_tile, W_L0))
   ```
   **【思考】**：这看起来很吓人。但实际上，你只要把它想成把 `OH` 维度、`OW` 维度、`OC` 维度转成一维标号。公式就是：`Z_index * (X的长度 * Y的长度) + Y_index * (X的长度) + X_index` 的经典降维打击。

---

## 🎯 给新手的全盘通关与 Debug 建议

你一定在写这堆填空时体会到了被张量下标支配的恐惧。送上三点通关锦囊：

1. **别光看，画出来**：很多 Bug 源于 `ceil_div`（向上取整）没有对齐。遇到 `slice_pad` 不断引发数组越界的报错时，找张草稿纸画一个简单的 $5 \times 5$ 表格，令 `stride=2`、原图 `pad=1`，手套公式验证你的 `slice_pad_r` 是不是多加了或少减了 `1`。
2. **控制变量隔离除错**：只要把 `step4.py` 测试用例中随机生成的 `pad` 写死成 0、`stride` 写进死成 1、`dilation` 写死成 1。如果跑不通，说明问题出在最基础的 L1 -> L0 坐标取样；如果跑通了而放开随机数后失败，那 Bug 100% 在上面那段四处寻找 `slice_pad` 悬空的 if/else 逻辑中。
3. **通过此项目的蜕变**：一旦你走通了这四个步骤，你对于 AI 底层那些形似天书的矩阵计算库，诸如 Cutlass，或者一些芯片 SDK 文档里描述的数据排布（`NC1HWC0` 之类）都不再会感到陌生。因为你亲手完成了一个 TPU 中最高难度的特征搬起与数据流控制。
