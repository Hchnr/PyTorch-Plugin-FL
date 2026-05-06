# 算子实现进度

## 已注册算子 (PrivateUse1 dispatch key)

以下算子已在 `Register.cpp` 中通过 `m.impl()` 注册。

### 基础设施 OPs (Factory / Copy / View)

| Op | wrapper | 说明 |
|----|---------|------|
| aten.empty.memory_format | wrapper_empty_memory_format | ✅ |
| aten.empty_strided | wrapper_empty_strided | ✅ |
| aten.as_strided | wrapper_as_strided | ✅ |
| aten.resize_ | wrapper_resize_ | ✅ |
| aten._reshape_alias | wrapper__reshape_alias | ✅ |
| aten._copy_from | wrapper__copy_from | ✅ |
| aten._copy_from_and_resize | wrapper__copy_from_and_resize | ✅ |
| aten._local_scalar_dense | wrapper__local_scalar_densor | ✅ |
| aten.set_.source_Tensor | wrapper_set_source_Tensor_ | ✅ |
| aten.set_.source_Storage | wrapper_set_source_Storage_ | ✅ |
| aten.set_.source_Storage_storage_offset | wrapper_set_source_Storage_storage_offsetset_ | ✅ |
| aten.view | wrapper_view | ✅ |
| aten.contiguous | wrapper_contiguous | ✅ (+ AutogradPrivateUse1) |
| aten.clone | wrapper_clone | ✅ |
| aten._to_copy | wrapper__to_copy | ✅ |
| aten.record_stream | wrapper_record_stream | ✅ |

### Compute OPs (通过 DispatchStub 分发)

| Op | stub | wrapper | 状态 |
|----|------|---------|------|
| aten.mm.default | mm_stub | wrapper_mm / wrapper_mm_out | ✅ 已完成 |
| aten.bmm.default | bmm_stub | wrapper_bmm / wrapper_bmm_out | ✅ 已完成 |
| aten.cat.default | cat_stub | wrapper_cat | ✅ 已完成 |
| aten.embedding.default | embedding_stub | wrapper_embedding | ✅ 已完成 |

---

## 待实现算子 (Infer 所需)

### 有 FlagGems C++ API — 可接入 DispatchStub

这些算子在 `liboperators.so` 中有导出符号，可以像 mm/bmm 一样接入。

| Op | FlagGems C++ 函数 | 状态 |
|----|-------------------|------|
| aten.addmm.default | `flag_gems::addmm` | ⬜ TODO |
| aten.softmax.int | `flag_gems::softmax` | ⬜ TODO |
| aten.argmax.default | `flag_gems::argmax` | ⬜ TODO |
| aten.sum.dim_IntList | `flag_gems::sum_dim` | ⬜ TODO |
| aten.topk.default | `flag_gems::topk` | ⬜ TODO |
| aten.nonzero.default | `flag_gems::nonzero` | ⬜ TODO |
| aten.copy_.default | `flag_gems::copy_` | ⬜ TODO |

### 无 FlagGems C++ API — Python-only Triton

这些算子在 FlagGems 中只有 Python Triton 实现，`liboperators.so` 无对应符号。
当前走 CPU fallback，后续可通过 `torch.library` 从 Python 侧注册。

| Op | 状态 |
|----|------|
| aten.add.Tensor | ⬜ CPU fallback |
| aten.mul.Tensor | ⬜ CPU fallback |
| aten.cos.default | ⬜ CPU fallback |
| aten.sin.default | ⬜ CPU fallback |
| aten.neg.default | ⬜ CPU fallback |
| aten.rsqrt.default | ⬜ CPU fallback |
| aten.silu.default | ⬜ CPU fallback |
| aten.pow.Tensor_Scalar | ⬜ CPU fallback |
| aten.mean.dim | ⬜ CPU fallback |
| aten.all.default | ⬜ CPU fallback |
| aten._scaled_dot_product_flash_attention_for_cpu.default | ⬜ CPU fallback |
| aten.arange.default | ⬜ CPU fallback |

### View/Metadata OPs — 走 CPU fallback 即可

纯 view 操作，不涉及数据搬运。

| Op | 状态 |
|----|------|
| aten._unsafe_view.default | ✅ fallback |
| aten.alias.default | ✅ fallback |
| aten.expand.default | ✅ fallback |
| aten.slice.Tensor | ✅ fallback |
| aten.t.default | ✅ fallback |
| aten.transpose.int | ✅ fallback |
| aten.unsqueeze.default | ✅ fallback |
| aten.lift_fresh.default | ✅ fallback |
