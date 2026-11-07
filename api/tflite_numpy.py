import struct
from typing import Dict, List, Optional

import numpy as np

#коды встроенных операций TFLite
_OPS = {0: "ADD", 1: "AVERAGE_POOL_2D", 3: "CONV_2D", 4: "DEPTHWISE_CONV_2D", 9: "FULLY_CONNECTED", 14: "LOGISTIC",
        17: "MAX_POOL_2D", 18: "MUL", 19: "RELU", 21: "RELU6", 22: "RESHAPE", 25: "SOFTMAX", 34: "PAD", 40: "MEAN", 41: "SUB"}
_DTYPES = {0: np.float32, 2: np.int32, 4: np.int64}#код типа данных TFLite -> numpy-тип


class _Table:

    def __init__(self, buf: bytes, pos: int):
        self.buf, self.pos = buf, pos

    def _field(self, i: int) -> int:
        vt = self.pos - struct.unpack_from("<i", self.buf, self.pos)[0]#vtable лежит ДО самой таблицы, отступ со знаком
        n = struct.unpack_from("<H", self.buf, vt)[0]#размер vtable в байтах
        o = 4 + 2 * i#первые 4 байта vtable — служебные, дальше по 2 байта на поле
        if o >= n:
            return 0#поля нет в схеме — считаем отсутствующим
        r = struct.unpack_from("<H", self.buf, vt + o)[0]#смещение поля относительно начала таблицы
        return self.pos + r if r else 0

    def scalar(self, i, fmt, default=0):
        p = self._field(i)
        return struct.unpack_from("<" + fmt, self.buf, p)[0] if p else default

    def table(self, i):
        p = self._field(i)
        return _Table(self.buf, p + struct.unpack_from("<I", self.buf, p)[0]) if p else None#uoffset: цель = адрес_ссылки + значение

    def _vec(self, i):
        p = self._field(i)
        if not p:
            return None
        v = p + struct.unpack_from("<I", self.buf, p)[0]#переходим по uoffset к самой векторной структуре
        return v + 4, struct.unpack_from("<I", self.buf, v)[0]#первые 4 байта вектора — длина, данные сразу за ними

    def scalars(self, i, fmt) -> list:
        r = self._vec(i)
        if r is None:
            return []
        st, n = r
        return list(struct.unpack_from(f"<{n}{fmt}", self.buf, st)) if n else []

    def tables(self, i) -> list:
        r = self._vec(i)
        if r is None:
            return []
        st, n = r
        return [_Table(self.buf, st + 4 * k + struct.unpack_from("<I", self.buf, st + 4 * k)[0]) for k in range(n)]#каждый элемент — uoffset на дочернюю таблицу

    def raw(self, i) -> bytes:
        r = self._vec(i)
        if r is None:
            return b""
        st, n = r
        return self.buf[st:st + n]#сырые байты вектора (используется для буферов весов)


def _act(x: np.ndarray, code: int) -> np.ndarray:
    if code == 0:
        return x
    if code == 1:
        return np.maximum(x, 0)
    if code == 3:
        return np.clip(x, 0, 6)
    raise NotImplementedError(f"fused activation {code}")


def _pads(in_size: int, k: int, stride: int, same: bool):
    if not same:
        return (in_size - k) // stride + 1, 0, 0
    out = -(-in_size // stride)#ceil-деление без math.ceil
    total = max((out - 1) * stride + k - in_size, 0)
    return out, total // 2, total - total // 2#при нечётном total больше добиваем справа/снизу


def conv2d(x, w, b, sh, sw, same, dh=1, dw=1):
    if dh != 1 or dw != 1:
        raise NotImplementedError("dilation != 1")
    n, h, wd, c = x.shape
    co, kh, kw, ci = w.shape#формат весов TFLite: (out_channels, kh, kw, in_channels)
    oh, pt, pb = _pads(h, kh, sh, same)
    ow, pl, pr = _pads(wd, kw, sw, same)
    if kh == 1 and kw == 1:#частный случай 1x1 — быстрее через матричное умножение без im2col
        xs = x[:, ::sh, ::sw, :][:, :oh, :ow, :]#шагаем по пикселям и обрезаем до точного размера выхода
        out = xs.reshape(-1, c) @ w.reshape(co, ci).T
        return (out + b).reshape(n, oh, ow, co).astype(np.float32)
    xp = np.pad(x, ((0, 0), (pt, pb), (pl, pr), (0, 0)))
    win = np.lib.stride_tricks.sliding_window_view(xp, (kh, kw), axis=(1, 2))[:, ::sh, ::sw][:, :oh, :ow]#окна без копирования памяти
    cols = win.transpose(0, 1, 2, 4, 5, 3).reshape(n * oh * ow, kh * kw * c)#im2col: (N,OH,OW,kh,kw,C) -> (N*OH*OW, kh*kw*C)
    out = cols @ w.reshape(co, kh * kw * ci).T + b
    return out.reshape(n, oh, ow, co).astype(np.float32)


def depthwise_conv2d(x, w, b, sh, sw, same, mult=1):

    if mult != 1:
        raise NotImplementedError("depth_multiplier != 1")
    n, h, wd, c = x.shape
    _, kh, kw, _ = w.shape#веса depthwise: (1, kh, kw, C) — по фильтру на канал
    oh, pt, pb = _pads(h, kh, sh, same)
    ow, pl, pr = _pads(wd, kw, sw, same)
    xp = np.pad(x, ((0, 0), (pt, pb), (pl, pr), (0, 0)))
    out = np.zeros((n, oh, ow, c), dtype=np.float32)
    for i in range(kh):
        for j in range(kw):
            out += xp[:, i:i + sh * (oh - 1) + 1:sh, j:j + sw * (ow - 1) + 1:sw, :] * w[0, i, j, :]#сдвигаем окно и умножаем на вес этого тапа
    return out + b


def _pool(x, kh, kw, sh, sw, same, fn):
    n, h, wd, c = x.shape
    oh, pt, pb = _pads(h, kh, sh, same)
    ow, pl, pr = _pads(wd, kw, sw, same)
    fill = -np.inf if fn is np.max else 0.0#для max-pool добивка не должна побеждать реальные значения
    xp = np.pad(x, ((0, 0), (pt, pb), (pl, pr), (0, 0)), constant_values=fill)
    win = np.lib.stride_tricks.sliding_window_view(xp, (kh, kw), axis=(1, 2))[:, ::sh, ::sw][:, :oh, :ow]
    return fn(win, axis=(4, 5)).astype(np.float32)#оси 4,5 — это kh и kw внутри окна


class Interpreter:
    def __init__(self, model: bytes):
        root = _Table(model, struct.unpack_from("<I", model, 0)[0])#в flatbuffer в начале файла uoffset на корневую таблицу
        codes = [max(oc.scalar(0, "b"), oc.scalar(3, "i", 0)) for oc in root.tables(1)]#builtin_code (byte) или deprecated_builtin_code (int32)
        buffers = [b.raw(0) for b in root.tables(4)]
        g = root.tables(2)[0]#берём первую (и обычно единственную) подграф-модель
        self.tensors = []
        for t in g.tables(0):
            dt = _DTYPES.get(t.scalar(1, "b", 0))
            if dt is None:
                raise NotImplementedError("поддерживаются только float32/int32 тензоры (для квантованной модели нужен официальный рантайм)")
            shape = t.scalars(0, "i")
            data = buffers[t.scalar(2, "I", 0)]#индекс в таблицу буферов, где лежат веса
            const = np.frombuffer(data, dtype=dt).reshape(shape).copy() if data else None#копия — иначе массив будет только для чтения
            self.tensors.append({"shape": shape, "dtype": dt, "const": const})
        self.inputs, self.outputs = g.scalars(1, "i"), g.scalars(2, "i")
        self.ops = []
        for o in g.tables(3):
            code = codes[o.scalar(0, "I", 0)]#opcode_index ссылается на таблицу OperatorCode
            name = _OPS.get(code)
            if name is None:
                raise NotImplementedError(f"операция TFLite №{code} не поддерживается встроенным интерпретатором")
            self.ops.append({"name": name, "in": o.scalars(1, "i"), "out": o.scalars(2, "i"), "opt": o.table(4)})

    @property
    def input_shape(self) -> List[int]:
        return list(self.tensors[self.inputs[0]]["shape"])

    def run(self, x: np.ndarray) -> np.ndarray:
        v: Dict[int, np.ndarray] = {self.inputs[0]: x.astype(np.float32)}#словарь «индекс тензора -> значение» на этот прогон
        get = lambda i: v[i] if i in v else self.tensors[i]["const"]#если не посчитано — значит это константа
        for op in self.ops:
            n, ins, opt = op["name"], op["in"], op["opt"]
            a = get(ins[0])
            bias = lambda: get(ins[2]) if len(ins) > 2 and ins[2] >= 0 else 0#индекс -1 означает «bias отсутствует»
            if n in ("ADD", "SUB", "MUL"):
                b = get(ins[1])
                r = a + b if n == "ADD" else a - b if n == "SUB" else a * b
                r = _act(r, opt.scalar(0, "b", 0) if opt else 0)
            elif n == "LOGISTIC":
                r = 0.5 * np.tanh(0.5 * a) + 0.5#численно устойчивый сигмоид без переполнения exp
            elif n == "RELU":
                r = np.maximum(a, 0)
            elif n == "RELU6":
                r = np.clip(a, 0, 6)
            elif n == "CONV_2D":
                r = conv2d(a, get(ins[1]), bias(), opt.scalar(2, "i", 1), opt.scalar(1, "i", 1), opt.scalar(0, "b", 0) == 0,
                           opt.scalar(5, "i", 1), opt.scalar(4, "i", 1))#padding: 0 == SAME, 1 == VALID
                r = _act(r, opt.scalar(3, "b", 0))
            elif n == "DEPTHWISE_CONV_2D":
                r = depthwise_conv2d(a, get(ins[1]), bias(), opt.scalar(2, "i", 1), opt.scalar(1, "i", 1),
                                     opt.scalar(0, "b", 0) == 0, opt.scalar(3, "i", 1))
                r = _act(r, opt.scalar(4, "b", 0))
            elif n == "MEAN":
                axes = tuple(int(i) for i in get(ins[1]).ravel())#оси приходят как тензор, а не как атрибут
                r = a.mean(axis=axes, keepdims=bool(opt.scalar(0, "B", 0)) if opt else False).astype(np.float32)
            elif n == "FULLY_CONNECTED":
                w = get(ins[1])#веса FC в TFLite лежат как (out, in)
                r = a.reshape(-1, w.shape[1]) @ w.T + bias()
                r = _act(r, opt.scalar(0, "b", 0) if opt else 0).astype(np.float32)
            elif n == "SOFTMAX":
                beta = opt.scalar(0, "f", 1.0) if opt else 1.0
                e = np.exp((a - a.max(axis=-1, keepdims=True)) * beta)#вычитаем максимум для численной устойчивости
                r = (e / e.sum(axis=-1, keepdims=True)).astype(np.float32)
            elif n == "RESHAPE":
                r = a.reshape(self.tensors[op["out"][0]]["shape"])#целевую форму берём из описания выходного тензора
            elif n in ("AVERAGE_POOL_2D", "MAX_POOL_2D"):
                r = _pool(a, opt.scalar(4, "i", 1), opt.scalar(3, "i", 1), opt.scalar(2, "i", 1), opt.scalar(1, "i", 1),
                          opt.scalar(0, "b", 0) == 0, np.mean if n == "AVERAGE_POOL_2D" else np.max)
                r = _act(r, opt.scalar(5, "b", 0))
            else:  # PAD
                p = get(ins[1]).astype(int)#матрица (rank, 2): пары (before, after) на каждую ось
                r = np.pad(a, [(int(lo), int(hi)) for lo, hi in p])
            v[op["out"][0]] = r
        return v[self.outputs[0]]