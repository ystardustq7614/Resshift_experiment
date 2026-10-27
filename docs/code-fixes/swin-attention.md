# Swin 注意力掩码与分支修正

更新日期：2026-10-27。状态：R001 已完成源码修复及局部验收，提交 `1ddc516` 已同步到 dev 并复测通过。下文原错误片段和问题行号保留为修复依据，不代表修复后的现状；当前证据见 [CHANGELOG](../../CHANGELOG.md)。

[返回代码问题索引](README.md) · [返回方案索引](../README.md)

本文件只维护掩码布局、token 顺序、非移位分支及对应局部验收；网络尺寸方案见 [model.md](../model.md)。

## R001 当前结果

Gemini 已修复 NCHW 区域索引、多余 token 转置、非移位／缓存／动态分支及名义尺寸类型。Codex 复跑 13 个测试全部通过，并独立用真实 Git 旧源码复现缺陷、对四层完整矩形尺寸逐元素比较几何参考、验证 CPU／CUDA BF16 的真实 BasicLayer 前后向和 mask device／dtype。该结果仅覆盖注意力局部正确性，不代表 U-Net 尺寸贯通、检查点、完整训练或目标双卡已通过。

## 两处掩码布局错误

**位置：** [calculate_mask():214](../../261023_Leizheng_resshift/models/swin_transformer.py)；窗口划分见同文件第 35～47 行，实际特征展平见第 258～259 行。

### 问题是什么

第一处，函数创建 `[1,1,H,W]` 的 NCHW 张量，却执行：

```python
img_mask[:, h, w, :] = cnt
```

这里 `h` 错落到 channel 轴，`w` 错落到 H 轴，W 轴未被正确分区。

第二处，函数随后执行：

```python
mask_windows = window_partition(img_mask, self.window_size).permute(0, 2, 3, 1).contiguous()
```

`window_partition()` 已返回 `[nW, Ws, Ws, 1]`。再次置换后展平，会交换窗口内 H/W token 顺序；实际特征展平没有执行这次交换。因此，只修第一处索引仍会把时间边界和道轴边界的限制用错位置。

### 需要怎么改

保留已有区域分段、编号和窗口划分函数，同时改正两处布局：

```python
# calculate_mask() 内保留现有 H/W、h_slices、w_slices 和 cnt 循环。
img_mask[:, :, h, w] = cnt

# window_partition 已返回 [nW, Ws, Ws, 1]，不再 permute。
mask_windows = window_partition(img_mask, self.window_size)
mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)
```

该函数仅供移位块使用；非移位块按下文的分支方案直接传 `mask=None`。不更改 `window_partition()`、`window_reverse()` 的布局，也不通过修改真实特征展平方式来迁就错误掩码。

### 已有证据与验收要求

真实函数 CPU 内存复现条件为 `H=16,W=32,window_size=8,shift_size=4`：

| 比较 | 不同元素数 |
| --- | ---: |
| 原实现 → 仅修 NCHW 索引 | 3072 |
| 仅修索引 → 同时删除多余 permute | 8192 |
| 原实现 → 两处同时修复 | 9216 |

第一行最后一个时间边界窗口，即窗口索引 3：

| 窗口内 token 对 | 仅修索引 | 正确要求 |
| --- | ---: | ---: |
| `(0,0)` 与 `(0,4)`，跨时间首尾 | 0，错误允许 | -100，屏蔽 |
| `(0,0)` 与 `(4,0)`，合法道间关联 | -100，错误屏蔽 | 0，允许 |

两个候选的 shape 都为 `[8,64,64]`，该窗口的屏蔽位置数都为 2048。原方案曾记录的“3072 处变化”和“2048 个屏蔽位置”只能证明索引修改改变了掩码，不能证明修复正确。

落盘后的验收必须检查具体 token 对，并与独立参考掩码逐元素比较，分别覆盖时间首尾、道轴首尾、合法相邻区域和内部窗口。再覆盖完整输入的各层矩形尺寸及实际选用的裁窗尺寸。

## 非移位分支与尺寸比较

**位置：** [SwinTransformerBlock 构造器:207](../../261023_Leizheng_resshift/models/swin_transformer.py)、[forward():262](../../261023_Leizheng_resshift/models/swin_transformer.py)、[PatchEmbed:473](../../261023_Leizheng_resshift/models/swin_transformer.py)。

### 问题与触发条件

- `shift_size=0` 时构造器注册 `attn_mask=None`，但尺寸匹配的 forward 分支直接执行 `self.attn_mask.to(...)`，会对 None 调用方法。
- 原实现的 `patches_resolution` 是列表，forward 的 `x_size` 是元组，数值相同也比较不相等。因此原常用路径可能一直动态生成掩码，遮蔽上述异常；不能将历史缺陷描述为原路径必然报错。R001 已统一类型并修复分支。
- 尺寸不匹配分支不先判断是否移位，会为普通窗口生成不需要的掩码；动态分支可能生成全零掩码，不能仅据此断言当前普通窗口数值一定错误。

### 需要怎么改

在构造时将名义分辨率统一为 `(H,W)` 元组，forward 同样使用元组；随后先判断移位，再选择掩码：

```python
actual_hw = (Ph, Pw)
nominal_hw = tuple(self.input_resolution)
if self.shift_size == 0:
    mask = None
elif actual_hw == nominal_hw:
    mask = self.attn_mask.to(device=x.device, dtype=x.dtype)
else:
    mask = self.calculate_mask(actual_hw).to(device=x.device, dtype=x.dtype)
attn_windows = self.attn(x_windows, mask=mask)
```

同时落实上一节的完整 `calculate_mask()` 修复。窗口尺寸和 shift 大小的有效性按方案入口约束检查，动态裁窗不得复用其他真实尺寸的掩码。

验收覆盖 shift=0/4、名义尺寸匹配/不匹配四类组合：非移位块始终传 None；移位块的缓存和动态掩码在同一真实尺寸下逐元素一致；CPU/GPU 和 AMP 下 device、dtype 均符合注意力运算要求。
