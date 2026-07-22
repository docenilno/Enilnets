"""Convolution variants for the graph API (roadmap item 35): standard,
strided, dilated, grouped, depthwise, separable, and causal convolution.

The whole family is ONE composite: an im2col-style patch **gather**
(integer-array indexing -- `getitem`, whose backward scatter-adds
duplicates from overlapping windows) followed by a batched **matmul**,
plus slicing/concat for groups. No convolution-specific gradient code
exists here at all -- exactly the "write the forward formula as graph ops"
promise of Phase 3. The `nn/` conv layers are untouched.
"""

from typing import Any, Optional, Tuple

from ..core.backend import np
from ..nn.weight_init import init_conv_weights, init_conv1d_weights
from .tensor import Tensor, as_tensor
from .layers import Layer, Parameter
from . import ops


def _pair(v: Any) -> Tuple[int, int]:
    return (int(v), int(v)) if isinstance(v, int) else (int(v[0]), int(v[1]))


def _out_size(n: int, k: int, stride: int, dilation: int) -> int:
    eff = dilation * (k - 1) + 1
    if n < eff:
        raise ValueError(
            f"input size {n} smaller than effective kernel {eff} "
            f"(k={k}, dilation={dilation}) -- pad the input first")
    return (n - eff) // stride + 1


def _conv2d_single_group(x: Tensor, w: Tensor, stride, dilation) -> Tensor:
    """Core composite: (B, C, H, W) x (OC, C, kh, kw) -> (B, OC, OH, OW)."""
    B, C, H, W = (int(s) for s in x.shape)
    OC, _, kh, kw = (int(s) for s in w.shape)
    sh, sw = stride
    dh, dw = dilation
    OH, OW = _out_size(H, kh, sh, dh), _out_size(W, kw, sw, dw)

    # Patch membership indices, host math on the active backend:
    # channel/row/col of every (C*kh*kw) patch element at every (OH*OW)
    # output position.
    c_idx = np.repeat(np.arange(C), kh * kw)[:, None]                # (Ckk, 1)
    off_h = np.tile(np.repeat(np.arange(kh) * dh, kw), C)[:, None]   # (Ckk, 1)
    off_w = np.tile(np.tile(np.arange(kw) * dw, kh), C)[:, None]     # (Ckk, 1)
    base_h = np.repeat(np.arange(OH) * sh, OW)[None, :]              # (1, L)
    base_w = np.tile(np.arange(OW) * sw, OH)[None, :]                # (1, L)

    cols = x[:, c_idx, off_h + base_h, off_w + base_w]               # (B, Ckk, L)
    out = ops.matmul(ops.reshape(w, shape=(OC, C * kh * kw)), cols)  # (B, OC, L)
    return ops.reshape(out, shape=(B, OC, OH, OW))


def conv2d(x: Any, weight: Any, bias: Optional[Any] = None, stride: Any = 1,
           padding: Any = 0, dilation: Any = 1, groups: int = 1) -> Tensor:
    """2-D convolution: ``x`` (B, IC, H, W), ``weight``
    (OC, IC/groups, kh, kw), optional ``bias`` (OC,).

    ``padding`` is symmetric zero-padding (int or (ph, pw)); ``stride``/
    ``dilation`` are ints or pairs. ``groups`` splits channels:
    ``groups=IC`` with OC a multiple of IC is depthwise convolution."""
    x, weight = as_tensor(x), as_tensor(weight)
    B, IC = int(x.shape[0]), int(x.shape[1])
    OC = int(weight.shape[0])
    stride, dilation = _pair(stride), _pair(dilation)
    ph, pw = _pair(padding)
    if IC % groups or OC % groups:
        raise ValueError(
            f"groups={groups} must divide both in_channels ({IC}) and "
            f"out_channels ({OC})")
    if int(weight.shape[1]) != IC // groups:
        raise ValueError(
            f"weight expects {int(weight.shape[1])} channels/group but input "
            f"has {IC // groups} (IC={IC}, groups={groups})")
    if ph or pw:
        x = ops.pad(x, pad_width=((0, 0), (0, 0), (ph, ph), (pw, pw)),
                    mode="constant")
    if groups == 1:
        out = _conv2d_single_group(x, weight, stride, dilation)
    else:
        icg, ocg = IC // groups, OC // groups
        pieces = [
            _conv2d_single_group(x[:, g * icg:(g + 1) * icg],
                                 weight[g * ocg:(g + 1) * ocg],
                                 stride, dilation)
            for g in range(groups)
        ]
        out = ops.concatenate(*pieces, axis=1)
    if bias is not None:
        out = ops.add(out, ops.reshape(as_tensor(bias), shape=(OC, 1, 1)))
    return out


def conv1d(x: Any, weight: Any, bias: Optional[Any] = None, stride: int = 1,
           padding: int = 0, dilation: int = 1, groups: int = 1) -> Tensor:
    """1-D convolution: ``x`` (B, IC, L), ``weight`` (OC, IC/groups, k) --
    same composite as :func:`conv2d` via a width-1 spatial axis."""
    x, weight = as_tensor(x), as_tensor(weight)
    B, IC, L = (int(s) for s in x.shape)
    OC, ICg, k = (int(s) for s in weight.shape)
    out = conv2d(ops.reshape(x, shape=(B, IC, L, 1)),
                 ops.reshape(weight, shape=(OC, ICg, k, 1)),
                 bias=bias, stride=(stride, 1), padding=(padding, 0),
                 dilation=(dilation, 1), groups=groups)
    OL = int(out.shape[2])
    return ops.reshape(out, shape=(B, OC, OL))


def causal_conv1d(x: Any, weight: Any, bias: Optional[Any] = None,
                  dilation: int = 1) -> Tensor:
    """Causal 1-D convolution (WaveNet-style): output[t] depends only on
    inputs at positions <= t, via left-only zero padding of
    ``dilation * (k - 1)``; output length equals input length."""
    weight = as_tensor(weight)
    k = int(weight.shape[2])
    left = dilation * (k - 1)
    x = ops.pad(as_tensor(x), pad_width=((0, 0), (0, 0), (left, 0)),
                mode="constant")
    return conv1d(x, weight, bias=bias, dilation=dilation)


def _triple(v: Any) -> Tuple[int, int, int]:
    return (int(v),) * 3 if isinstance(v, int) else tuple(int(i) for i in v)


def _conv3d_single_group(x: Tensor, w: Tensor, stride, dilation) -> Tensor:
    """(B, C, D, H, W) x (OC, C, kd, kh, kw) -> (B, OC, OD, OH, OW): the
    same gather+matmul composite as 2-D, one more spatial axis."""
    B, C, D, H, W = (int(s) for s in x.shape)
    OC, _, kd, kh, kw = (int(s) for s in w.shape)
    sd, sh, sw = stride
    dd, dh, dw = dilation
    OD = _out_size(D, kd, sd, dd)
    OH = _out_size(H, kh, sh, dh)
    OW = _out_size(W, kw, sw, dw)
    K = kd * kh * kw

    kd_i, kh_i, kw_i = np.meshgrid(np.arange(kd) * dd, np.arange(kh) * dh,
                                   np.arange(kw) * dw, indexing="ij")
    c_idx = np.repeat(np.arange(C), K)[:, None]
    off_d = np.tile(kd_i.reshape(-1), C)[:, None]
    off_h = np.tile(kh_i.reshape(-1), C)[:, None]
    off_w = np.tile(kw_i.reshape(-1), C)[:, None]
    od_i, oh_i, ow_i = np.meshgrid(np.arange(OD) * sd, np.arange(OH) * sh,
                                   np.arange(OW) * sw, indexing="ij")
    base_d = od_i.reshape(-1)[None, :]
    base_h = oh_i.reshape(-1)[None, :]
    base_w = ow_i.reshape(-1)[None, :]

    cols = x[:, c_idx, off_d + base_d, off_h + base_h, off_w + base_w]
    out = ops.matmul(ops.reshape(w, shape=(OC, C * K)), cols)
    return ops.reshape(out, shape=(B, OC, OD, OH, OW))


def conv3d(x: Any, weight: Any, bias: Optional[Any] = None, stride: Any = 1,
           padding: Any = 0, dilation: Any = 1, groups: int = 1) -> Tensor:
    """3-D convolution: ``x`` (B, IC, D, H, W), ``weight``
    (OC, IC/groups, kd, kh, kw) -- the dimensional generalization of
    :func:`conv2d`, same options."""
    x, weight = as_tensor(x), as_tensor(weight)
    IC, OC = int(x.shape[1]), int(weight.shape[0])
    stride, dilation = _triple(stride), _triple(dilation)
    pd, ph, pw = _triple(padding)
    if IC % groups or OC % groups:
        raise ValueError(
            f"groups={groups} must divide both in_channels ({IC}) and "
            f"out_channels ({OC})")
    if pd or ph or pw:
        x = ops.pad(x, pad_width=((0, 0), (0, 0), (pd, pd), (ph, ph), (pw, pw)),
                    mode="constant")
    if groups == 1:
        out = _conv3d_single_group(x, weight, stride, dilation)
    else:
        icg, ocg = IC // groups, OC // groups
        out = ops.concatenate(*[
            _conv3d_single_group(x[:, g * icg:(g + 1) * icg],
                                 weight[g * ocg:(g + 1) * ocg],
                                 stride, dilation)
            for g in range(groups)], axis=1)
    if bias is not None:
        out = ops.add(out, ops.reshape(as_tensor(bias), shape=(OC, 1, 1, 1)))
    return out


def _zero_stuff(x: Tensor, axis: int, stride: int) -> Tensor:
    """Insert ``stride - 1`` zeros between consecutive elements along
    ``axis`` (length n -> (n-1)*stride + 1) -- the upsampling half of a
    transposed convolution, built from concat/reshape/slice ops."""
    if stride == 1:
        return x
    shape = tuple(int(s) for s in x.shape)
    n = shape[axis]
    expanded = ops.reshape(x, shape=shape[:axis + 1] + (1,) + shape[axis + 1:])
    zeros_shape = shape[:axis + 1] + (stride - 1,) + shape[axis + 1:]
    zeros = Tensor(np.zeros(zeros_shape, dtype=x.dtype))
    stuffed = ops.concatenate(expanded, zeros, axis=axis + 1)
    stuffed = ops.reshape(stuffed, shape=shape[:axis] + (n * stride,) + shape[axis + 1:])
    index = tuple([slice(None)] * axis + [slice(0, (n - 1) * stride + 1)])
    return stuffed[index]


def _flip(w: Tensor, axes: Tuple[int, ...]) -> Tensor:
    index = [slice(None)] * w.ndim
    for ax in axes:
        index[ax] = slice(None, None, -1)
    return w[tuple(index)]


def conv_transpose2d(x: Any, weight: Any, bias: Optional[Any] = None,
                     stride: Any = 1, padding: Any = 0,
                     output_padding: Any = 0) -> Tensor:
    """2-D transposed ("de")convolution: the adjoint of :func:`conv2d`
    with the same stride/padding, so ``<conv(x, w), y> ==
    <x, conv_transpose(y, w)>`` (pinned by tests). ``weight`` uses the
    transposed-conv convention (IC, OC, kh, kw); output spatial size is
    ``(n-1)*stride - 2*padding + k + output_padding``.

    Implemented as zero-stuffing by ``stride``, padding by ``k-1-padding``
    (+ ``output_padding`` bottom/right), then a stride-1 conv with the
    spatially-flipped, channel-transposed kernel -- all existing ops."""
    x, weight = as_tensor(x), as_tensor(weight)
    sh, sw = _pair(stride)
    ph, pw = _pair(padding)
    oph, opw = _pair(output_padding)
    kh, kw = int(weight.shape[2]), int(weight.shape[3])
    if oph >= sh or opw >= sw:
        raise ValueError(
            f"output_padding {(oph, opw)} must be smaller than stride {(sh, sw)}")
    up = _zero_stuff(_zero_stuff(x, 2, sh), 3, sw)
    up = ops.pad(up, pad_width=((0, 0), (0, 0),
                                (kh - 1 - ph, kh - 1 - ph + oph),
                                (kw - 1 - pw, kw - 1 - pw + opw)),
                 mode="constant")
    kernel = ops.transpose(_flip(weight, (2, 3)), axes=(1, 0, 2, 3))
    return conv2d(up, kernel, bias=bias)


def conv_transpose1d(x: Any, weight: Any, bias: Optional[Any] = None,
                     stride: int = 1, padding: int = 0,
                     output_padding: int = 0) -> Tensor:
    """1-D transposed convolution: ``x`` (B, IC, L), ``weight``
    (IC, OC, k) -- rides on :func:`conv_transpose2d` via a width-1 axis."""
    x, weight = as_tensor(x), as_tensor(weight)
    B, IC, L = (int(s) for s in x.shape)
    _, OC, k = (int(s) for s in weight.shape)
    out = conv_transpose2d(ops.reshape(x, shape=(B, IC, L, 1)),
                           ops.reshape(weight, shape=(IC, OC, k, 1)),
                           bias=bias, stride=(stride, 1), padding=(padding, 0),
                           output_padding=(output_padding, 0))
    return ops.reshape(out, shape=(B, OC, int(out.shape[2])))


def conv_transpose3d(x: Any, weight: Any, bias: Optional[Any] = None,
                     stride: Any = 1, padding: Any = 0,
                     output_padding: Any = 0) -> Tensor:
    """3-D transposed convolution: ``x`` (B, IC, D, H, W), ``weight``
    (IC, OC, kd, kh, kw) -- same construction as 2-D, one more axis."""
    x, weight = as_tensor(x), as_tensor(weight)
    sd, sh, sw = _triple(stride)
    pd, ph, pw = _triple(padding)
    opd, oph, opw = _triple(output_padding)
    kd, kh, kw = (int(s) for s in weight.shape[2:])
    if opd >= sd or oph >= sh or opw >= sw:
        raise ValueError("output_padding must be smaller than stride")
    up = _zero_stuff(_zero_stuff(_zero_stuff(x, 2, sd), 3, sh), 4, sw)
    up = ops.pad(up, pad_width=((0, 0), (0, 0),
                                (kd - 1 - pd, kd - 1 - pd + opd),
                                (kh - 1 - ph, kh - 1 - ph + oph),
                                (kw - 1 - pw, kw - 1 - pw + opw)),
                 mode="constant")
    kernel = ops.transpose(_flip(weight, (2, 3, 4)), axes=(1, 0, 2, 3, 4))
    return conv3d(up, kernel, bias=bias)


class Conv2D(Layer):
    """Graph conv layer covering the whole item-35 family via its
    arguments: ``dilation`` for dilated, ``groups`` for grouped,
    ``groups=in_ch`` for depthwise."""

    def __init__(self, in_ch: int, out_ch: int, k: Any, stride: Any = 1,
                 padding: Any = 0, dilation: Any = 1, groups: int = 1,
                 use_bias: bool = True, init_method: str = "he_normal") -> None:
        super().__init__()
        kh, kw = _pair(k)
        if kh != kw:
            raise ValueError("Conv2D currently uses square kernels (kh == kw)")
        w, b = init_conv_weights(in_ch // groups, out_ch, kh, method=init_method)
        self.weight = Parameter(w, name="weight")
        self.bias = Parameter(b, name="bias") if use_bias else None
        self.stride, self.padding, self.dilation, self.groups = \
            stride, padding, dilation, groups

    def forward(self, x: Tensor) -> Tensor:
        return conv2d(x, self.weight, bias=self.bias, stride=self.stride,
                      padding=self.padding, dilation=self.dilation,
                      groups=self.groups)


class Conv1D(Layer):
    """1-D counterpart of :class:`Conv2D`; ``causal=True`` switches to
    left-only padding (output[t] sees only inputs <= t)."""

    def __init__(self, in_ch: int, out_ch: int, k: int, stride: int = 1,
                 padding: int = 0, dilation: int = 1, groups: int = 1,
                 causal: bool = False, use_bias: bool = True,
                 init_method: str = "he_normal") -> None:
        super().__init__()
        if causal and (stride != 1 or padding != 0):
            raise ValueError("causal Conv1D implies stride=1, padding=0 "
                             "(it pads left by dilation*(k-1) itself)")
        w, b = init_conv1d_weights(in_ch // groups, out_ch, k, method=init_method)
        self.weight = Parameter(w, name="weight")
        self.bias = Parameter(b, name="bias") if use_bias else None
        self.stride, self.padding, self.dilation = stride, padding, dilation
        self.groups, self.causal = groups, causal

    def forward(self, x: Tensor) -> Tensor:
        if self.causal:
            return causal_conv1d(x, self.weight, bias=self.bias,
                                 dilation=self.dilation)
        return conv1d(x, self.weight, bias=self.bias, stride=self.stride,
                      padding=self.padding, dilation=self.dilation,
                      groups=self.groups)


class SeparableConv2D(Layer):
    """Depthwise-separable convolution: a depthwise ``k x k`` conv
    (``groups=in_ch``, ``depth_multiplier`` filters per channel) followed
    by a pointwise ``1 x 1`` conv mixing channels -- the
    MobileNet/Xception building block."""

    def __init__(self, in_ch: int, out_ch: int, k: int, stride: Any = 1,
                 padding: Any = 0, depth_multiplier: int = 1,
                 use_bias: bool = True, init_method: str = "he_normal") -> None:
        super().__init__()
        self.depthwise = Conv2D(in_ch, in_ch * depth_multiplier, k,
                                stride=stride, padding=padding, groups=in_ch,
                                use_bias=False, init_method=init_method)
        self.pointwise = Conv2D(in_ch * depth_multiplier, out_ch, 1,
                                use_bias=use_bias, init_method=init_method)

    def forward(self, x: Tensor) -> Tensor:
        return self.pointwise(self.depthwise(x))
