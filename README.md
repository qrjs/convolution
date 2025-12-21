# 使用说明
为了让初学者循序渐进地理解矩阵分块、im2col、卷积切分，以及data layout，在此给出简单的上手代码。请新人参考[OpenTPU的切分策略](https://fcnzwpvwu9lm.feishu.cn/wiki/SDCpwp3zBijCvnkQZrBcHWzjnhc)，按照step1.py->step2->step3->step4的顺序调试代码，直至step4.py测试通过。

```
python3 step4.py 
{'batch': 1, 'in_height': 90, 'in_width': 63, 'out_height': 43, 'out_width': 12, 'in_channel': 48, 'out_channel': 50, 'kernel_h': 3, 'kernel_w': 6, 'stride_h': 2, 'stride_w': 5, 'pad_h': 2, 'pad_w': 2, 'dilation_h': 4, 'dilation_w': 2, 'CO_L1': 48, 'H_L1': 41, 'W_L1': 2, 'CI_L1': 31, 'CO_L0': 40, 'H_L0': 20, 'W_L0': 1, 'CI1_L0': 3}
pass
{'batch': 3, 'in_height': 28, 'in_width': 78, 'out_height': 12, 'out_width': 24, 'in_channel': 30, 'out_channel': 63, 'kernel_h': 3, 'kernel_w': 3, 'stride_h': 2, 'stride_w': 3, 'pad_h': 1, 'pad_w': 0, 'dilation_h': 3, 'dilation_w': 4, 'CO_L1': 52, 'H_L1': 2, 'W_L1': 18, 'CI_L1': 28, 'CO_L0': 40, 'H_L0': 1, 'W_L0': 13, 'CI1_L0': 2}
pass
{'batch': 1, 'in_height': 86, 'in_width': 56, 'out_height': 14, 'out_width': 11, 'in_channel': 86, 'out_channel': 19, 'kernel_h': 6, 'kernel_w': 2, 'stride_h': 5, 'stride_w': 5, 'pad_h': 0, 'pad_w': 0, 'dilation_h': 4, 'dilation_w': 4, 'CO_L1': 4, 'H_L1': 6, 'W_L1': 6, 'CI_L1': 43, 'CO_L0': 4, 'H_L0': 2, 'W_L0': 4, 'CI1_L0': 4}
pass
{'batch': 1, 'in_height': 79, 'in_width': 62, 'out_height': 25, 'out_width': 9, 'in_channel': 24, 'out_channel': 78, 'kernel_h': 2, 'kernel_w': 4, 'stride_h': 3, 'stride_w': 6, 'pad_h': 0, 'pad_w': 3, 'dilation_h': 4, 'dilation_w': 6, 'CO_L1': 56, 'H_L1': 3, 'W_L1': 8, 'CI_L1': 4, 'CO_L0': 24, 'H_L0': 2, 'W_L0': 4, 'CI1_L0': 1}
pass
{'batch': 1, 'in_height': 50, 'in_width': 53, 'out_height': 13, 'out_width': 7, 'in_channel': 6, 'out_channel': 72, 'kernel_h': 2, 'kernel_w': 6, 'stride_h': 4, 'stride_w': 6, 'pad_h': 1, 'pad_w': 0, 'dilation_h': 2, 'dilation_w': 3, 'CO_L1': 52, 'H_L1': 2, 'W_L1': 3, 'CI_L1': 5, 'CO_L0': 40, 'H_L0': 1, 'W_L0': 2, 'CI1_L0': 1}
pass
{'batch': 3, 'in_height': 53, 'in_width': 87, 'out_height': 11, 'out_width': 22, 'in_channel': 9, 'out_channel': 84, 'kernel_h': 2, 'kernel_w': 1, 'stride_h': 5, 'stride_w': 4, 'pad_h': 1, 'pad_w': 0, 'dilation_h': 4, 'dilation_w': 4, 'CO_L1': 12, 'H_L1': 2, 'W_L1': 17, 'CI_L1': 3, 'CO_L0': 8, 'H_L0': 1, 'W_L0': 5, 'CI1_L0': 1}
pass
{'batch': 2, 'in_height': 39, 'in_width': 46, 'out_height': 26, 'out_width': 7, 'in_channel': 58, 'out_channel': 9, 'kernel_h': 4, 'kernel_w': 5, 'stride_h': 1, 'stride_w': 6, 'pad_h': 1, 'pad_w': 1, 'dilation_h': 5, 'dilation_w': 2, 'CO_L1': 4, 'H_L1': 16, 'W_L1': 2, 'CI_L1': 15, 'CO_L0': 4, 'H_L0': 11, 'W_L0': 1, 'CI1_L0': 3}
pass
{'batch': 3, 'in_height': 7, 'in_width': 37, 'out_height': 3, 'out_width': 33, 'in_channel': 25, 'out_channel': 68, 'kernel_h': 6, 'kernel_w': 2, 'stride_h': 3, 'stride_w': 1, 'pad_h': 3, 'pad_w': 0, 'dilation_h': 1, 'dilation_w': 4, 'CO_L1': 28, 'H_L1': 2, 'W_L1': 17, 'CI_L1': 22, 'CO_L0': 20, 'H_L0': 1, 'W_L0': 3, 'CI1_L0': 1}
pass
```

# 阶段目标
## step1
1. 参考[OpenTPU的切分策略](https://tju-opentpu.feishu.cn/wiki/SDCpwp3zBijCvnkQZrBcHWzjnhc)，理解矩阵如何分块
2. 参考[Data Layout简介](https://tju-opentpu.feishu.cn/wiki/LU0jwZIN7iEBhHke5bbccqegnle)理解什么是M1K1M0K0，N1K1N0K0，M1N1M0N0

## step2
1. 填充`MK2K1MK0_DDR2L1`, 理解layout变换
2. 参考`matmul_mk_kn`，理解bias和deq操作，填充`matmul_m1k1m0k0_n1k1n0k0`
3. 参考[OpenTPU的切分策略](https://tju-opentpu.feishu.cn/wiki/SDCpwp3zBijCvnkQZrBcHWzjnhc)，理解bias_en, psum_en, deq_en的条件

## step3
1. 参考[Convolution和Im2col详解](https://tju-opentpu.feishu.cn/wiki/S2wCwVIdgiQnl2kviwXcRdc2nqc)理解im2col如何实现

## step4
1. 参考`MK2K1MK0_DDR2L1`，填充`BHWC2C1BHWC0_DDR2L1`，理解C1BHWC0与K1MK0的对应关系
2. 填充`C1BHWC02M1K1M0K0_L12L0`，巩固对im2col的理解
3. 填充pad_t, pad_b, pad_l, pad_r，理解切分时的补0策略
4. 参考step3 bias_en, psum_en, deq_en的条件，巩固对卷积的理解