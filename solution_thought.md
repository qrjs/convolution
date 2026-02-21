# TPU 卷积运算模拟：从矩阵分块到完整卷积代码解析与通关指南

> [!NOTE]
> 本文档基于本项目 `step1.py` - `step4.py` 的**填空修改记录**，为你详细拆解整个项目的逻辑主线。在这个项目中，我们通过四个渐进式的 Python 脚本模拟了底层 AI 硬件（比如 NPU/TPU）处理矩阵乘法和卷积的过程。
> 
> 下面的讲解不仅结合了代码中填补的**关键公式与循环逻辑**，还给出了各个环节在出现 Bug 时的**具体报错表象**与**排错排查思路**。

---

## 📖 全局视角与硬件对齐理念

在看具体的长串多维循环前，先在脑海中建立这个物理模型：
1. **DDR（大仓库）**：数据以最线性的方式存放，比如普通的矩阵 `(M, K)` 或是图像数据 `(B, H, W, C)`。
2. **L1 Buffer（片上缓存）**：由于 DDR 读取太慢，我们要把数据搬运到 L1 中。为了方便之后极速地被计算核心吞吐，数据要被重组成一种包含**分块参数**的多维排布，例如 `(M1, K1, M0, K0)`。
3. **L0 Register / MAC 阵列（计算引擎）**：计算单元每次只能消化极小固定尺寸的矩阵块，本例中定义的架构参数为 `M0=3, N0=4, K0=5`。

你的四个 `step` 脚本就是不断在实现：**如何正确切割 -> 搬运 -> 组合下标运算 -> 调用基础矩阵乘核心 -> 将结果写回**。

---

## 🛠️ Step 1: 矩阵分块与阵列核心计算模拟

**目标**：理解 `M1K1M0K0` 布局结构的意义，以及硬件计算单元是如何以小卡片 `(M0, N0)` 为单位进行矩阵乘法的。

### 核心代码拆解
在 `step1.py` 中，主要完成的是**核心循环累加**与**坐标系拉平**：

1. **底层小核心计算逻辑**：
   在 TPU/NPU 内部，不可能一次吃下一个 $1000 \times 1000$ 的大矩阵，而是嵌套了外部分块循环 (`m1, n1`) 和内部累加循环 (`k1`)。
   ```python
   for m1 in range(M1):
       for n1 in range(N1):
           # 针对当前 (M1, N1) 分块，初始化一个小方格累加器 temp (相当于局部寄存器)
           temp = np.zeros((M0, N0)) 
           for k1 in range(K1):
               # 核心单元计算: L(M0xK0) 乘以 R转置(K0xN0)，累加到 temp 的(M0xN0)里
               temp += left_m1k1m0k0[m1][k1] @ right_n1k1n0k0[n1][k1].T
               
           # 跑完了所有的 k1 累加，当前这块结果写回 SRAM 缓存
           result_m1n1m0n0[m1][n1] = temp
   ```
   *注意：代码里的 `@` 符号代表着底层电路级的一条 `MAC`（乘加）指令的宏观表现。*

2. **全局结果逆向重组与去 Padding**：
   切分时因为使用了 `ceil_div`（向上取整对齐到 $M_0, N_0$），边缘会产生废数据（0）。写回普通矩阵 `result_mn` 时需要截断：
   ```python
   # 解析出的全局物理坐标：大块数 * 每块长度 + 块内偏移量
   m = m1 * M0 + m0
   n = n1 * N0 + n0
   
   # 防越界保护：必须把向外取整补的零 (Padding) 抛弃掉
   if m < M and n < N:
       result_mn[m][n] = result_m1n1m0n0[m1][n1][m0][n0]
   ```

---

## 🛠️ Step 2: 内存排布与控制信号（全剧最关键流控）

**目标**：理解从 `MK` 变成 `K1MK0` 的 Layout 映射公式，以及控制状态机里的 `bias`, `psum`, `deq` 触发时机。这段逻辑如果在芯片设计里写错，硬件会直接发生内存崩坏或计算错乱。

### 核心代码拆解
1. **DDR 到 L1 (将平铺二维变成多维排布)**：
   在 `MK2K1MK0_DDR2L1` 填空：
   ```python
   # 计算原图 DDR 的一维线性绝对坐标 K
   # k1 代表在第几个分块，K0 是块宽，k0 是块内第几个元素
   k = start_k + k1 * K0 + k0
   
   # 只有在有效数据范围内，才从 DDR(matrix_mk) 抠数据给到 SRAM(matrix_k1mk0)
   if k < start_k + tile_k and k < tensor_k:
       matrix_k1mk0[k1][m][k0] = matrix_mk[start_m + m][k]
   ```

2. **硬件控制信号的逻辑推演 (极易出错点)**：
   在巨大的外层循环里，硬件何时加载 Bias，何时累加，何时释放反量化输出？
   ```python
   # bias_en：偏置加上去只能加一次。
   # 只有当这是整个矩阵K维度的第一次大循环(tile_k) 的第一小分片(slice_k) 时，才为真。
   bias_en = (tile_k_start_in_tensor == 0 and slice_k_start_in_tile == 0)
   
   # psum_en：部分和使能(Partial Sum)。
   # 除了第一次计算，后面的所有切片循环，计算引擎产出的结果都必须跟缓存里的上次结果累加！
   psum_en = (tile_k_start_in_tensor != 0 or slice_k_start_in_tile != 0)
   
   # deq_en：反量化。一定放在所有加法做完之后。
   # 即当前 Tile 内走到最后一块的尽头，且当前全局的 Tile 也是最后一个 K。
   is_last_k_in_tile = (slice_k_start_in_tile + k_size_slice >= k_size_tile_align_k0)
   is_last_tile_in_k = (tile_k_start_in_tensor + k_size_tile >= TENSOR_K)
   deq_en = is_last_k_in_tile and is_last_tile_in_k
   ```

### 🐞 Debug 经验 - Step 2 报错解析：
如果你发现 `step2.py` 的测试用例跑出 `Fail`（金数据核对不上）：
- **现象 A: 输出结果比预期大了很多，甚至翻倍。** $\rightarrow$ 重点查 `bias_en`。一定是你的 `bias_en` 在后面的循环里也被错误激活成了 `True`，导致加了多次偏置。
- **现象 B: 输出结果严重不符，像丢了数据。** $\rightarrow$ 重点查 `psum_en`。很可能是你丢了上一轮的部分和（没有写回或者覆盖了）。

---

## 🛠️ Step 3: Im2Col（卷积转矩阵的灵魂映射）

**目标**：理解如何彻底消灭 4 重卷积滑动窗口循环，转为一次干净利落的大型矩阵排布。

### 核心代码拆解
在 `CHW2MK` 中，你需要把图片 `(Channel, Height, Width)` 填成 `(M, K)` 形式。这里你填了全场最重要的坐标映射系：

```python
for kh in range(kernel_h):
    for kw in range(kernel_w):
        # 核心映射：由当前输出图坐标 oh/ow 与滑动核座标 kh/kw，反推原图有效提取点 ih/iw
        # 考虑的因素：(1) Stride 放大输出距离 (2) Padding 导致起始点向负方向左移 (3) Dilation 导致核内吸附像素点散开
        ih = oh * stride_h - pad_h + kh * dilation_h
        iw = ow * stride_w - pad_w + kw * dilation_w
        
        # 必须过滤掉打在外层 Padding 环境(负数或超图片范围) 上的光束
        if (0 <= ih < in_height) and (0 <= iw < in_width):
            for ic in range(in_channel):
                # 利用 K 维度一维展平法则：ic权重最大，次之是 kh，最小是 kw
                k_idx = ic * kernel_h * kernel_w + kh * kernel_w + kw
                ofm[m_idx][k_idx] = ifm[ic][ih][iw]
```
跑通 `im2col`，这意味着卷积问题正式被降维成了上面的 `Step1/2` 矩阵乘法问题。

---

## 🛠️ Step 4: 大整合与动态 Padding 边界计算 (压轴 Boss 战)

**目标**：把 `im2col` 理论融入多重切分大 Tile 嵌套环境里面去。你将踩遍所有越界的坑。

### 核心代码拆解与调试心得
在 `test_convolution()` 中有六层的嵌套循环，其中**最容易算死人的是对动态切片 `slice_pad_r/b` 的推断**：

1. **为什么切分出的大 Tile 还要重算边界 Padding?**
   原始的 `pad_h` 和 `pad_w` 参数是糊在屏幕整张图片上的。当我们把图片切成了 20 个格子，**位于图片中央的格子在取卷积窗数据时，是绝对碰不到补零的！**
   只有最贴右边、最贴下边的格子，它的卷积核伸到了图片虚无外场，才需要补零。这就叫**局部 Padding 动态计算**。
   
   **你填补的下边补零 (`slice_pad_b`) 思路**：
   ```python
   # 当这个切分块覆盖的终点在加上整个核（含有膨胀因子）覆盖距离后，超出了图片最底下边界 in_height
   if(slice_oh_end_in_tensor * stride_h - pad_h + dilation_h * (kernel_h - 1) >= in_height):
       # 那么超出悬空的这段差值，就是这个切片需要局域补充的 padding bottom
       slice_pad_b = (slice_oh_end_in_tensor * stride_h - pad_h + dilation_h * (kernel_h - 1)) - (in_height - 1)
   else:
       slice_pad_b = 0  # 没触底，就在图片内部，padding为0
   ```

2. **一维内存编号运算**：
   ```python
   psb_addr = (slice_oh_start_in_tile//H_L0 * ceil_div(ow_size_tile, W_L0) + slice_ow_start_in_tile//W_L0 + \
               slice_oc_start_in_tile//CO_L0 * ceil_div(oh_size_tile, H_L0) * ceil_div(ow_size_tile, W_L0))
   ```
   **逻辑拆解**：把它看成一个扁平化公式：`Z层数 * (当前层总容量：X总长度 * Y总长度) + Y排数 * (每一排的X容量：X总长度) + X柱子偏移`。这里 `X` 是块宽，`Y` 是块高，`Z` 对应通道维度。

### 🐞 Step 4 究极 Debug 指南：被列表越界支配的恐惧
如果你在 `step4.py` 中看到了满屏红色的 `IndexError: index X is out of bounds for axis Y`，按照以下步骤排摸：

- **第一步：锁死变量降维打击**。绝对不要硬着头皮去瞪六层外循环。
  - 把用例生成器的参数改死：`stride_h=1`, `pad_h=0`, `dilation=1`。
  - 如果连这种最简用例也出 `IndexError` 报错，说明你的 `L1` -> `L0` 的绝对坐标推导崩了。去找 `m_index = b * slice_oh * slice_ow + oh * slice_ow + ow` 这行，仔细对对乘的系数。

- **第二步：纸笔画图打印核对**。
  - 如果只有带了 `Padding` 和 `Stride=2` 时才会崩，报错多半指向在生成 `ifm_m1k1m0k0` 的数组构建位置。
  - 原因很简单，你通过那堆复杂的 `slice_pad_b` 算出的边界有偏差。
  - **解决方案**：在这个代码位置插两个 `print`。
    ```python
    print(f"当前块起点: {slice_oh_start_in_tensor}, 核算pad_b为: {slice_pad_b}")
    ```
    拿一张纸，画一个 $6 \times 6$ 大小的二维网格充当原图，把当前起点的坐标落笔点在格子上。用核大小 `kernel_h=3`, 步长 `stride=2` 往前画一个框。你肉眼看到超出格子的长度，如果和你 `print` 的 `slice_pad_b` 不一样（例如相差了 1 ），那么就意味着由于 `>=` 或 `-1` 边界闭区间没处理好，你的算法写错了。

> **终极结语**
> 走通这个步骤不仅让你熟练运用 Numpy 张量魔法，更是逼你切换回底层电子工程师的视角：一切高深复杂的运算网络，在硬件寄存器里不过就是一群小蚂蚁排布着极其精妙的列阵，精准地踏着 `m` 和 `k` 的步点，在正确的时钟信号触发下进出存储器罢了。
