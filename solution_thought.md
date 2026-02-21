# TPU 卷积运算模拟：从矩阵分块到完整卷积代码解析与通关指南

> [!NOTE]
> 本文档基于本项目 `step1.py` - `step4.py` 的**填空修改记录**，为你详细拆解整个项目的逻辑主线。在这个项目中，我们通过四个渐进式的 Python 脚本模拟了底层 AI 硬件（比如 NPU/TPU）处理矩阵乘法和卷积的过程。
> 
> 下面的讲解不仅结合了**代码 Diff 修改前后对比**，还详实地给出了各个环节在出现 Bug 时的**具体报错表象**与**排错排查思路**，强烈建议配合源码对读。

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

### 核心代码 Diff：硬件矩阵乘法核心填空
原来模版里只是简单的循环外壳，我们需要补全硬件的核心计算逻辑。

```diff
-            temp = np.zeros((M0, N0))
+            # 针对当前 M1, N1 切片，初始化小方格累加器 temp (相当于局部寄存器)
+            temp = np.zeros((M0, N0)) 
             for k1 in range(K1):
-
+                # 核心单元计算: L(M0xK0) 乘以 R转置(K0xN0)，由于 K 被切成 K1 份，在 temp 中累加
+                temp += left_m1k1m0k0[m1][k1] @ right_n1k1n0k0[n1][k1].T
+
+            # 跑完了所有的 k1 累加，当前这块结果写回 SRAM 缓存
             result_m1n1m0n0[m1][n1] = temp
```
*注意：代码里的 `@` 符号代表着底层电路级的一条 `MAC`（乘加阵列）指令的宏观表现。*

### 核心代码 Diff：全局结果提取与去 Padding
因为使用了 `ceil_div`（向上取整对齐到 $M_0, N_0$），边缘不可避免地补了 0 (Padding)。写回结果 `result_mn` 时我们需要摒弃这些空气数据。

```diff
-    result_mn = result_m1m0n1n0
+    # 将分块结果转换回线性排布 (M1N1M0N0 -> MN)
+    result_mn = np.zeros((M, N))
+    for m1 in range(M1):
+        for n1 in range(N1):
+            for m0 in range(M0):
+                for n0 in range(N0):
+                    # 计算出全局的绝对物理坐标
+                    m = m1 * M0 + m0
+                    n = n1 * N0 + n0
+                    # 防越界保护：必须把向外取整补的零剔除
+                    if m < M and n < N:
+                        result_mn[m][n] = result_m1n1m0n0[m1][n1][m0][n0]
```

---

## 🛠️ Step 2: 内存排布与控制信号（全剧最关键流控）

**目标**：理解控制状态机里的 `bias`, `psum`, `deq` 触发时机。这段逻辑如果在芯片设计里写错，硬件会直接发生结果爆炸或严重丢表。

### 核心代码 Diff：硬件控制信号生成 (极易出错点)
在宏大的外层循环里，硬件何时加载 Bias，何时打开累加器（Accumulator），何时开始释放反量化输出（Dequantization）？

```diff
-                            bias_en = 
-                            psum_en = 
-                            deq_en = 
+                            # 控制信号生成
+                            # 1. 只有在当前 Tile 绝对的第一次计算时，才启用加偏置
+                            bias_en = (tile_k_start_in_tensor == 0 and slice_k_start_in_tile == 0)
+                            
+                            # 2. 部分和使能(Partial Sum)。除了第一次计算，后面都需要累加之前的分块结果
+                            psum_en = (tile_k_start_in_tensor != 0 or slice_k_start_in_tile != 0)
+                            
+                            # 3. 反量化。只有当整个矩阵该块的 K 维度被全部遍历到底后，才做反量化。
+                            is_last_k_in_tile = (slice_k_start_in_tile + k_size_slice >= k_size_tile_align_k0)
+                            is_last_tile_in_k = (tile_k_start_in_tensor + k_size_tile >= TENSOR_K)
+                            deq_en = is_last_k_in_tile and is_last_tile_in_k
```

### 🐞 Debug指南 - Step 2 报错解析：
如果你发现你的程序输出 `Fail` (结果不对)：
- **现象 A (数值翻倍跳跃):** 重点查 `bias_en`。一定是你在后续计算中没有关闭它，导致一个区域的值加了多次偏置！正确思路是：此通道 K=0 时才允许开启。
- **现象 B (数值被拦截、缺失):** 重点查 `psum_en`。初学者极易填成 `psum_en = True` 或者忘了判断第一次计算，导致最后输出全错。

---

## 🛠️ Step 3: Im2Col（卷积转矩阵的灵魂映射）

**目标**：理解如何彻底消灭 4 重卷积滑动窗口嵌套，转存为可交给巨型矩阵乘法器的一维排布矩阵。

### 核心代码 Diff
在 `CHW2MK` 源码中，你需要算出每个窗口抓取的 `(ih, iw)` 在哪里：

```diff
             for kh in range(kernel_h):
                 for kw in range(kernel_w):
-                    ih = 
-                    iw = 
-                    if (0 <= ih < in_height) and (0 <= iw < in_width):
-                        for ic in range(in_channel):
+                    # 计算当前卷积核坐标对应输入原图上的坐标
+                    ih = oh * stride_h - pad_h + kh * dilation_h
+                    iw = ow * stride_w - pad_w + kw * dilation_w
+                    
+                    # 过滤掉打在外层 Padding 环境(负数或超范围) 上的光点
+                    if (0 <= ih < in_height) and (0 <= iw < in_width):
+                        for ic in range(in_channel):
+                            # 一维展平：计算在 K 维度长条里的索引号
+                            k_idx = ic * kernel_h * kernel_w + kh * kernel_w + kw
+                            ofm[m_idx][k_idx] = ifm[ic][ih][iw]
```
跑通这一步，接下来的问题就全都转为了 Step1 与 2 的普通数学矩阵乘法任务。

---

## 🛠️ Step 4: 大整合与动态 Padding 边界计算 (压轴 Boss 战)

**目标**：把理论的 `im2col` 真正融入大块 Tile 嵌套的存储体系（`BCHW -> C1BHWC0 -> M1K1M0K0`）。这里是数组越界报错的重灾区。

### 核心代码 Diff：局域 Padding 界定
最初由于每次都把全切片视为普通情况，其实处于图片角落的格子会有“悬空”。你需要计算出这些角落块到底需要补零多少段。

```diff
                                     if(slice_oh_start_in_tensor * stride_h - pad_h < 0):
-                                        slice_pad_t = 
+                                        # 向上越界补偿
+                                        slice_pad_t = pad_h - slice_oh_start_in_tensor * stride_h
                                     else:
                                         slice_pad_t = 0
-                                    if():
-                                        slice_pad_b = 
-                                    else:
-                                        slice_pad_b = 0
+                                    
+                                    # 当格子尽头探测范围超出了原图最底端
+                                    if(slice_oh_end_in_tensor * stride_h - pad_h + dilation_h * (kernel_h - 1) >= in_height):
+                                        # 超出悬空的这段差值，就是这个切片需要补零的 padding bottom
+                                        slice_pad_b = (slice_oh_end_in_tensor * stride_h - pad_h + dilation_h * (kernel_h - 1)) - (in_height - 1)
+                                    else:
+                                        slice_pad_b = 0
```
*(左右局域边界 `pad_l`, `pad_r` 代码与此完全对称同理，可去源码处查看)*

### 🐞 究极 Debug 指南：被列表越界支配的恐惧
如果你在 `step4.py` 中看到了满屏红色的 `IndexError: index X is out of bounds for axis Y`，一定按照这几个口诀抢救：

1. **口诀一：锁死变量降维打击**。绝对不要硬着头皮去瞪六层外循环的 Log 排错。
   - 打开 `step4.py` 把原本随机的参数锁死配置：`stride_h=1`, `pad_h=0`, `dilation=1`。
   - 如果连这种最简白板参数也会跑出 `IndexError` 报错，说明你的 `L1` -> `L0` 的绝对坐标大公式彻底写错了（去查 `m_index = b * slice_oh * slice_ow + ...` 的相乘系数）。

2. **口诀二：纸笔画图，大胆 Print**。
   - 如果你只有在 `Padding=1` 和 `Stride=2` 开启时程序才会崩溃，八成就是上面那堆边界 `slice_pad_b/r` 的 `< / <= / +1 / -1` 弄出差错了。
   - **唯一排解法**：去代码处打印 `print(f"起点:{slice_oh_start_in_tensor}, 算得边界pad为:{slice_pad_b}")`。
   - 拿笔在纸上画个 $6 \times 6$ 的小格子模拟图片，用 $3 \times 3$ 步长为 2 的矩形框自己推一下，看看你肉眼推断的“悬空越出界外”的值跟你程序里公式打印出的值差了多少！往往就是一格宽度的修正错误。

> **学长结语**
> 不怕报错，怕的是不敢写 Print。走通这个填空矩阵项目，不仅让你在数值 Python 上炉火纯青，更是逼你切换回底层芯片设计工程师视角——因为这套循环在真实 TPU 编译器的 C++ Kernel 中，就是如同这般运行着。
