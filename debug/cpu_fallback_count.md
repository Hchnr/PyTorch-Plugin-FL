Inference: 94.28s, 59 tokens, 0.63 tok/s

======================================================================
CPU FALLBACK TRACE REPORT
======================================================================
Total op calls on flagos tensors: 191752
  Native (GPU):      110508 (57.6%)
  CPU fallback:      81244 (42.4%)

──────────────────────────────────────────────────────────────────────
FALLBACK OPS (sorted by call count):
──────────────────────────────────────────────────────────────────────
   25075x  aten::view
   14986x  aten::_unsafe_view
   11623x  aten::t
    8319x  aten::transpose.int
    6902x  aten::slice.Tensor
    6844x  aten::expand
    3599x  aten::unsqueeze
    3304x  aten::mul.Scalar
     177x  aten::select.int
     120x  aten::_local_scalar_dense
      59x  aten::lift_fresh
      59x  aten::new_ones
      59x  aten::multinomial
      59x  aten::squeeze.dim
      59x  aten::rsub.Scalar

──────────────────────────────────────────────────────────────────────
NATIVE OPS (sorted by call count):
──────────────────────────────────────────────────────────────────────
   21889x  aten::_to_copy
   21830x  aten::mul.Tensor
   13480x  aten::add.Tensor
   11623x  aten::mm
    6844x  aten::cat
    6667x  aten::pow.Tensor_Scalar
    6667x  aten::mean.dim
    6667x  aten::rsqrt
    3363x  aten::bmm
    3304x  aten::neg
    3304x  aten::repeat_interleave.self_int
    1652x  aten::_safe_softmax
    1652x  aten::silu
     145x  aten::clone
     119x  aten::masked_fill.Scalar
     118x  aten::_softmax
     118x  aten::__or__.Tensor
      60x  aten::isin.Tensor_Tensor
      60x  aten::cumsum
      60x  aten::eq.Scalar
      59x  aten::embedding
      59x  aten::all
      59x  aten::cos
      59x  aten::sin
      59x  aten::div.Tensor
      59x  aten::topk
      59x  aten::lt.Tensor
      59x  aten::sort
      59x  aten::le.Scalar
      59x  aten::copy_
      59x  aten::scatter.src
      59x  aten::bitwise_not
      59x  aten::bitwise_and.Tensor
      59x  aten::max
      28x  aten::tril
      28x  aten::where.self
       2x  aten::any
       1x  aten::lt.Scalar
       1x  aten::sub.Tensor
======================================================================