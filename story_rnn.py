#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
story_rnn.py — пословная рекуррентная нейронная сеть на чистом NumPy,
которая пишет не одну фразу, а целый рассказ.

Чем отличается от word_rnn.py
-----------------------------
1. Ячейка GRU (вентильная) вместо простой tanh-RNN. У обычной RNN память
   «затухает» через 15-20 слов, а рассказу нужно помнить героя и место
   действия десятки предложений. Вентили обновления и сброса дают сети
   явный механизм «держать» состояние сколь угодно долго.
2. Разметка рассказов служебными токенами <bos> / <eos>: сеть учится не
   просто языку, а форме — где завязка, где концовка, когда пора закончить.
3. Генерация целого текста: подсчёт предложений, разбиение на абзацы,
   nucleus-сэмплирование (top-p), штраф за повтор слов и запрет повторять
   n-граммы — без этого RNN на длинной дистанции сваливается в цикл.

Примеры запуска
---------------
  # обучение
  python story_rnn.py train --data stories.txt --epochs 40 --model story.npz

  # написать рассказ
  python story_rnn.py story --model story.npz --sentences 12 --out rasskaz.txt

  # рассказ по заданному началу
  python story_rnn.py story --model story.npz --prompt "Жил-был старый рыбак"

  # интерактивный режим: несколько рассказов подряд
  python story_rnn.py chat --model story.npz

Формат обучающего файла: обычный .txt. Если рассказы разделены пустой
строкой, сеть увидит границы и научится начинать и заканчивать текст.

Зависимости: numpy.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from collections import Counter

import numpy as np


UNK, BOS, EOS = "<unk>", "<bos>", "<eos>"
SPECIALS = (UNK, BOS, EOS)

_TOKEN_RE = re.compile(r"\w+(?:[-'’]\w+)*|[^\w\s]", re.UNICODE)
_NO_SPACE_BEFORE = set(".,!?;:%)]}»…\"'’")
_NO_SPACE_AFTER = set("([{«\"'“")
_SENTENCE_END = {".", "!", "?", "…"}


def tokenize(text: str, lowercase: bool = False) -> list[str]:
    if lowercase:
        text = text.lower()
    return _TOKEN_RE.findall(text)


def detokenize(tokens) -> str:
    """Склеивает токены в текст и приводит в порядок пробелы и заглавные буквы."""
    out = []
    for i, tok in enumerate(tokens):
        if i == 0:
            out.append(tok)
            continue
        prev = tokens[i - 1]
        if tok in _NO_SPACE_BEFORE or prev in _NO_SPACE_AFTER:
            out.append(tok)
        else:
            out.append(" " + tok)
    text = "".join(out)
    # заглавная буква в начале и после точки
    text = re.sub(r"(^|[.!?…]\s+)([а-яёa-z])",
                  lambda m: m.group(1) + m.group(2).upper(), text)
    return text


def read_corpus(path: str, lowercase: bool = False) -> list[str]:
    """
    Читает файл и превращает его в поток токенов, где каждый рассказ обёрнут
    в <bos> ... <eos>. Рассказы разделяются пустой строкой; если пустых строк
    в файле нет, весь текст считается одним рассказом.
    """
    with open(path, encoding="utf-8") as f:
        raw = f.read()
    chunks = [c.strip() for c in re.split(r"\n\s*\n", raw) if c.strip()]
    tokens: list[str] = []
    for chunk in chunks:
        tokens.append(BOS)
        tokens.extend(tokenize(chunk, lowercase=lowercase))
        tokens.append(EOS)
    return tokens


# ---------------------------------------------------------------------------
# Словарь
# ---------------------------------------------------------------------------
class Vocab:
    def __init__(self, words):
        self.itos = list(words)
        for sp in reversed(SPECIALS):
            if sp not in self.itos:
                self.itos.insert(0, sp)
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        self.unk_id = self.stoi[UNK]
        self.bos_id = self.stoi[BOS]
        self.eos_id = self.stoi[EOS]

    @classmethod
    def from_tokens(cls, tokens, min_freq: int = 1, max_size: int | None = None):
        counts = Counter(tokens)
        words = [w for w, c in counts.most_common()
                 if c >= min_freq and w not in SPECIALS]
        if max_size is not None:
            words = words[:max_size]
        return cls(words)

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, tokens) -> list[int]:
        return [self.stoi.get(w, self.unk_id) for w in tokens]

    def unknown(self, tokens) -> list[str]:
        return [w for w in tokens if w not in self.stoi]


# ---------------------------------------------------------------------------
# Ячейки
# ---------------------------------------------------------------------------
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


class VanillaCell:
    """h_t = tanh(Wx·x + Wh·h + b) — простая рекуррентная ячейка."""

    name = "rnn"

    def __init__(self, embed: int, hidden: int, rng):
        self.p = {
            "c_Wx": rng.standard_normal((hidden, embed)) / np.sqrt(embed),
            "c_Wh": rng.standard_normal((hidden, hidden)) / np.sqrt(hidden),
            "c_b": np.zeros((hidden, 1)),
        }

    def forward(self, x, h):
        hn = np.tanh(self.p["c_Wx"] @ x + self.p["c_Wh"] @ h + self.p["c_b"])
        return hn, (x, h, hn)

    def backward(self, dh, cache, g):
        x, h, hn = cache
        da = (1.0 - hn ** 2) * dh
        g["c_Wx"] += da @ x.T
        g["c_Wh"] += da @ h.T
        g["c_b"] += da
        return self.p["c_Wx"].T @ da, self.p["c_Wh"].T @ da


class GRUCell:
    """
    Вентильная ячейка GRU:

        z = σ(Wz·x + Uz·h + bz)          вентиль обновления: сколько взять нового
        r = σ(Wr·x + Ur·h + br)          вентиль сброса: сколько забыть старого
        ĥ = tanh(Wc·x + Uc·(r ⊙ h) + bc) кандидат нового состояния
        h' = (1 - z) ⊙ h + z ⊙ ĥ         смесь старого и нового

    Если z близко к нулю, состояние переносится дальше без изменений — так
    сеть удерживает героя и место действия на протяжении всего рассказа.
    """

    name = "gru"

    def __init__(self, embed: int, hidden: int, rng):
        def W(a, b):
            return rng.standard_normal((a, b)) / np.sqrt(b)

        self.p = {
            "c_Wz": W(hidden, embed), "c_Uz": W(hidden, hidden),
            "c_bz": np.zeros((hidden, 1)),
            "c_Wr": W(hidden, embed), "c_Ur": W(hidden, hidden),
            "c_br": np.zeros((hidden, 1)),
            "c_Wc": W(hidden, embed), "c_Uc": W(hidden, hidden),
            "c_bc": np.zeros((hidden, 1)),
        }

    def forward(self, x, h):
        p = self.p
        z = sigmoid(p["c_Wz"] @ x + p["c_Uz"] @ h + p["c_bz"])
        r = sigmoid(p["c_Wr"] @ x + p["c_Ur"] @ h + p["c_br"])
        rh = r * h
        c = np.tanh(p["c_Wc"] @ x + p["c_Uc"] @ rh + p["c_bc"])
        hn = (1.0 - z) * h + z * c
        return hn, (x, h, z, r, rh, c)

    def backward(self, dh, cache, g):
        p = self.p
        x, h, z, r, rh, c = cache

        dz = dh * (c - h)
        dc = dh * z
        dh_prev = dh * (1.0 - z)

        dc_raw = dc * (1.0 - c ** 2)
        g["c_Wc"] += dc_raw @ x.T
        g["c_Uc"] += dc_raw @ rh.T
        g["c_bc"] += dc_raw
        drh = p["c_Uc"].T @ dc_raw
        dr = drh * h
        dh_prev += drh * r

        dz_raw = dz * z * (1.0 - z)
        g["c_Wz"] += dz_raw @ x.T
        g["c_Uz"] += dz_raw @ h.T
        g["c_bz"] += dz_raw
        dh_prev += p["c_Uz"].T @ dz_raw

        dr_raw = dr * r * (1.0 - r)
        g["c_Wr"] += dr_raw @ x.T
        g["c_Ur"] += dr_raw @ h.T
        g["c_br"] += dr_raw
        dh_prev += p["c_Ur"].T @ dr_raw

        dx = (p["c_Wz"].T @ dz_raw + p["c_Wr"].T @ dr_raw + p["c_Wc"].T @ dc_raw)
        return dx, dh_prev


CELLS = {"rnn": VanillaCell, "gru": GRUCell}


# ---------------------------------------------------------------------------
# Модель
# ---------------------------------------------------------------------------
class StoryRNN:
    """Эмбеддинг слова → рекуррентная ячейка → softmax по словарю."""

    def __init__(self, vocab_size: int, hidden_size: int = 256,
                 embed_size: int = 96, cell: str = "gru", seed: int | None = None):
        rng = np.random.default_rng(seed)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.cell_name = cell
        self.cell = CELLS[cell](embed_size, hidden_size, rng)

        self.E = rng.standard_normal((embed_size, vocab_size)) * 0.01
        self.Why = rng.standard_normal((vocab_size, hidden_size)) / np.sqrt(hidden_size)
        self.by = np.zeros((vocab_size, 1))

    @property
    def params(self) -> dict[str, np.ndarray]:
        return {"E": self.E, "Why": self.Why, "by": self.by, **self.cell.p}

    def zero_state(self) -> np.ndarray:
        return np.zeros((self.hidden_size, 1))

    # -- обучение ------------------------------------------------------------
    def loss_and_grads(self, inputs, targets, h_prev):
        hs, ps, caches = {-1: h_prev.copy()}, {}, {}
        loss = 0.0

        for t, idx in enumerate(inputs):
            x = self.E[:, idx:idx + 1]
            hs[t], caches[t] = self.cell.forward(x, hs[t - 1])
            y = self.Why @ hs[t] + self.by
            y -= y.max()
            e = np.exp(y)
            ps[t] = e / e.sum()
            loss += -np.log(ps[t][targets[t], 0] + 1e-12)

        grads = {k: np.zeros_like(v) for k, v in self.params.items()}
        dh_next = np.zeros_like(h_prev)

        for t in reversed(range(len(inputs))):
            dy = ps[t].copy()
            dy[targets[t], 0] -= 1.0
            grads["Why"] += dy @ hs[t].T
            grads["by"] += dy

            dh = self.Why.T @ dy + dh_next
            dx, dh_next = self.cell.backward(dh, caches[t], grads)
            grads["E"][:, inputs[t]] += dx.ravel()

        n = max(len(inputs), 1)
        for g in grads.values():
            g /= n
            np.clip(g, -5.0, 5.0, out=g)

        return loss / n, grads, hs[len(inputs) - 1]

    # -- один шаг генерации --------------------------------------------------
    def step(self, idx: int, h):
        x = self.E[:, idx:idx + 1]
        h, _ = self.cell.forward(x, h)
        return (self.Why @ h + self.by).ravel(), h

    # -- сохранение / загрузка -----------------------------------------------
    def save(self, path: str, vocab: Vocab) -> None:
        np.savez_compressed(path, words=np.array(vocab.itos, dtype=object),
                            hidden_size=self.hidden_size,
                            embed_size=self.embed_size,
                            cell=self.cell_name, **self.params)

    @classmethod
    def load(cls, path: str):
        data = np.load(path, allow_pickle=True)
        vocab = Vocab(list(data["words"]))
        model = cls(len(vocab), int(data["hidden_size"]), int(data["embed_size"]),
                    cell=str(data["cell"]))
        model.E, model.Why, model.by = data["E"], data["Why"], data["by"]
        for k in model.cell.p:
            model.cell.p[k] = data[k]
        return model, vocab


# ---------------------------------------------------------------------------
# Выбор следующего слова
# ---------------------------------------------------------------------------
def pick_next(logits: np.ndarray, history: list[int], vocab: Vocab, rng, *,
              temperature: float = 0.9, top_k: int = 0, top_p: float = 0.92,
              repetition_penalty: float = 1.0, penalty_window: int = 40,
              no_repeat_ngram: int = 4, banned: set[int] | None = None) -> int:
    """
    Превращает выход сети в конкретное слово.

    На длинном тексте «жадный» или наивно случайный выбор одинаково плохи:
    первый зацикливается, второй несёт чушь. Поэтому:
      * temperature — сглаживает или заостряет распределение;
      * top-p (nucleus) — оставляет минимальный набор слов с суммарной
        вероятностью p и выбирает только из него;
      * repetition_penalty — понижает шанс слов, только что использованных;
      * no_repeat_ngram — прямо запрещает повторить n-грамму, которая в этом
        тексте уже была: главное лекарство от «сеть пошла по кругу».
    """
    logits = logits.astype(np.float64).copy()

    # служебные токены и <unk> никогда не печатаем (кроме явно разрешённого <eos>)
    logits[vocab.unk_id] = -np.inf
    logits[vocab.bos_id] = -np.inf
    if banned:
        for b in banned:
            logits[b] = -np.inf

    # штраф за недавние повторы (знаки препинания не трогаем)
    if repetition_penalty > 1.0 and history:
        for idx in set(history[-penalty_window:]):
            if vocab.itos[idx] in _NO_SPACE_BEFORE or vocab.itos[idx] in _NO_SPACE_AFTER:
                continue
            logits[idx] -= np.log(repetition_penalty)

    # запрет повторять уже встречавшиеся n-граммы
    if no_repeat_ngram > 1 and len(history) >= no_repeat_ngram - 1:
        prefix = tuple(history[-(no_repeat_ngram - 1):])
        for i in range(len(history) - no_repeat_ngram + 1):
            if tuple(history[i:i + no_repeat_ngram - 1]) == prefix:
                logits[history[i + no_repeat_ngram - 1]] = -np.inf

    logits = logits / max(temperature, 1e-6)
    if not np.isfinite(logits).any():
        return vocab.eos_id

    if top_k and top_k < len(logits):
        keep = np.argpartition(logits, -top_k)[-top_k:]
        mask = np.full_like(logits, -np.inf)
        mask[keep] = logits[keep]
        logits = mask

    p = np.exp(logits - np.max(logits))
    p /= p.sum()

    if 0.0 < top_p < 1.0:
        order = np.argsort(p)[::-1]
        cum = np.cumsum(p[order])
        cut = int(np.searchsorted(cum, top_p)) + 1
        keep = order[:cut]
        mask = np.zeros_like(p)
        mask[keep] = p[keep]
        s = mask.sum()
        if s > 0:
            p = mask / s

    return int(rng.choice(len(p), p=p))


# ---------------------------------------------------------------------------
# Генерация рассказа
# ---------------------------------------------------------------------------
def write_story(model: StoryRNN, vocab: Vocab, *, prompt: str = "",
                sentences: int = 12, max_words: int = 400,
                sentences_per_paragraph: int = 4, min_sentence_words: int = 4,
                lowercase: bool = False,
                rng: np.random.Generator | None = None, **sampling) -> str:
    """
    Пишет связный текст заданной длины и разбивает его на абзацы.

    Генерация идёт словами, но останавливается не по счётчику слов, а по
    числу законченных предложений — поэтому рассказ не обрывается на полуслове.
    """
    rng = rng or np.random.default_rng()
    h = model.zero_state()

    prompt_tokens = tokenize(prompt, lowercase=lowercase) if prompt.strip() else []
    seq = [vocab.bos_id] + vocab.encode(prompt_tokens)

    # прогреваем состояние началом рассказа
    for idx in seq[:-1]:
        _, h = model.step(idx, h)
    cur = seq[-1]

    # заранее соберём индексы знаков препинания — по ним удобно ставить запреты
    punct_ids = {i for i, w in enumerate(vocab.itos) if not w[0].isalnum()
                 and w not in SPECIALS}
    end_ids = {vocab.stoi[w] for w in _SENTENCE_END if w in vocab.stoi}

    produced = list(vocab.encode(prompt_tokens))
    # для показа держим исходные слова подсказки: даже если внутри они стали
    # <unk>, в готовом тексте пользователь должен видеть то, что написал
    words = list(prompt_tokens)
    n_sent = 0
    words_in_sentence = 0
    finished = False

    while n_sent < sentences and len(produced) < max_words:
        logits, h = model.step(cur, h)

        banned = set()
        # не даём закончить рассказ раньше, чем набрано нужное число предложений
        if n_sent < sentences - 1:
            banned.add(vocab.eos_id)
        # не даём ставить точку после двух-трёх слов
        if words_in_sentence < min_sentence_words:
            banned |= end_ids
        # два знака препинания подряд — почти всегда мусор
        if produced and produced[-1] in punct_ids:
            banned |= punct_ids

        cur = pick_next(logits, produced, vocab, rng, banned=banned, **sampling)

        if cur == vocab.eos_id:
            finished = True
            break
        produced.append(cur)
        words.append(vocab.itos[cur])
        if cur in end_ids:
            n_sent += 1
            words_in_sentence = 0
        elif cur not in punct_ids:
            words_in_sentence += 1

    # если текст оборвался на середине фразы — обрезаем до последней точки
    if not finished and words and words[-1] not in _SENTENCE_END:
        for i in range(len(words) - 1, -1, -1):
            if words[i] in _SENTENCE_END:
                words = words[:i + 1]
                break
        else:
            words.append(".")

    # разбивка на абзацы по числу предложений
    paragraphs, buf, count = [], [], 0
    for w in words:
        buf.append(w)
        if w in _SENTENCE_END:
            count += 1
            if count >= sentences_per_paragraph:
                paragraphs.append(detokenize(buf))
                buf, count = [], 0
    if buf:
        paragraphs.append(detokenize(buf))

    return "\n\n".join(p for p in paragraphs if p.strip())


# ---------------------------------------------------------------------------
# Обучение
# ---------------------------------------------------------------------------
def train(model: StoryRNN, vocab: Vocab, tokens, *, epochs: int = 40,
          seq_len: int = 30, lr: float = 0.002, decay: float = 0.95,
          log_every: int = 200, verbose: bool = True) -> StoryRNN:
    data = np.array(vocab.encode(tokens), dtype=np.int64)
    if len(data) < seq_len + 1:
        raise ValueError(f"Слишком мало текста: нужно хотя бы {seq_len + 1} слов.")

    mem = {k: np.zeros_like(v) for k, v in model.params.items()}
    smooth = np.log(len(vocab))
    step = 0
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        h = model.zero_state()
        for p in range(0, len(data) - seq_len - 1, seq_len):
            inputs = data[p:p + seq_len].tolist()
            targets = data[p + 1:p + seq_len + 1].tolist()
            loss, grads, h = model.loss_and_grads(inputs, targets, h)
            smooth = 0.999 * smooth + 0.001 * loss

            for name, param in model.params.items():          # RMSProp
                g = grads[name]
                mem[name] = decay * mem[name] + (1.0 - decay) * g * g
                param -= lr * g / (np.sqrt(mem[name]) + 1e-8)

            step += 1
            if verbose and log_every and step % log_every == 0:
                print(f"эпоха {epoch:>3}/{epochs}  шаг {step:>6}  "
                      f"ошибка {smooth:.3f}  перплексия "
                      f"{float(np.exp(min(smooth, 20))):7.1f}  "
                      f"({time.time() - t0:.0f} с)", flush=True)

    if verbose:
        print(f"Обучение завершено: {step} шагов, ошибка {smooth:.3f}, "
              f"{time.time() - t0:.0f} с.\n")
    return model


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def add_sampling_args(p):
    p.add_argument("--temperature", type=float, default=0.9, help="разнообразие")
    p.add_argument("--top-k", type=int, default=0, help="выбирать из k лучших слов (0 — выкл.)")
    p.add_argument("--top-p", type=float, default=0.92, help="nucleus-сэмплирование")
    p.add_argument("--repetition-penalty", type=float, default=1.0,
                   help="штраф за недавно использованные слова (1.0 — выкл.)")
    p.add_argument("--no-repeat-ngram", type=int, default=4,
                   help="запретить повтор n-грамм такой длины (0 — выкл.)")


def sampling_kwargs(a):
    return dict(temperature=a.temperature, top_k=a.top_k, top_p=a.top_p,
                repetition_penalty=a.repetition_penalty,
                no_repeat_ngram=a.no_repeat_ngram)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Пословная RNN/GRU на NumPy, которая пишет целый рассказ.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    tr = sub.add_parser("train", help="обучить модель на сборнике рассказов")
    tr.add_argument("--data", required=True, help="путь к .txt (рассказы через пустую строку)")
    tr.add_argument("--model", default="story.npz", help="куда сохранить модель")
    tr.add_argument("--cell", choices=list(CELLS), default="gru", help="тип ячейки")
    tr.add_argument("--hidden", type=int, default=256, help="размер скрытого слоя")
    tr.add_argument("--embed", type=int, default=96, help="размер эмбеддинга слова")
    tr.add_argument("--seq-len", type=int, default=30, help="длина фрагмента для BPTT")
    tr.add_argument("--epochs", type=int, default=40, help="число проходов по тексту")
    tr.add_argument("--lr", type=float, default=0.002, help="скорость обучения")
    tr.add_argument("--decay", type=float, default=0.95, help="сглаживание RMSProp")
    tr.add_argument("--min-freq", type=int, default=1, help="порог частоты слова")
    tr.add_argument("--max-vocab", type=int, default=None, help="ограничить словарь")
    tr.add_argument("--lowercase", action="store_true", help="привести текст к нижнему регистру")
    tr.add_argument("--seed", type=int, default=0, help="зерно случайных чисел")

    st = sub.add_parser("story", help="написать рассказ")
    st.add_argument("--model", default="story.npz", help="путь к модели")
    st.add_argument("--prompt", default="", help="начало рассказа (можно не задавать)")
    st.add_argument("--sentences", type=int, default=12, help="сколько предложений написать")
    st.add_argument("--paragraph", type=int, default=4, help="предложений в абзаце")
    st.add_argument("--max-words", type=int, default=400, help="жёсткий предел по словам")
    st.add_argument("--min-sentence-words", type=int, default=4,
                    help="не ставить точку раньше этого числа слов")
    st.add_argument("--out", default=None, help="сохранить рассказ в файл")
    st.add_argument("--lowercase", action="store_true", help="если модель обучена с --lowercase")
    st.add_argument("--seed", type=int, default=None, help="зерно для воспроизводимости")
    add_sampling_args(st)

    ch = sub.add_parser("chat", help="писать рассказы по очереди в диалоге")
    ch.add_argument("--model", default="story.npz", help="путь к модели")
    ch.add_argument("--sentences", type=int, default=12, help="сколько предложений писать")
    ch.add_argument("--paragraph", type=int, default=4, help="предложений в абзаце")
    ch.add_argument("--max-words", type=int, default=400, help="жёсткий предел по словам")
    ch.add_argument("--min-sentence-words", type=int, default=4,
                    help="не ставить точку раньше этого числа слов")
    ch.add_argument("--lowercase", action="store_true", help="если модель обучена с --lowercase")
    ch.add_argument("--seed", type=int, default=None, help="зерно для воспроизводимости")
    add_sampling_args(ch)
    return p


def load_or_die(path: str):
    if not os.path.exists(path):
        print(f"Модель не найдена: {path}. Сначала запустите "
              f"`python {os.path.basename(__file__)} train --data ваш.txt`",
              file=sys.stderr)
        raise SystemExit(1)
    return StoryRNN.load(path)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "train":
        if not os.path.exists(args.data):
            print(f"Файл не найден: {args.data}", file=sys.stderr)
            return 1
        tokens = read_corpus(args.data, lowercase=args.lowercase)
        if not tokens:
            print("В обучающем файле нет текста.", file=sys.stderr)
            return 1

        vocab = Vocab.from_tokens(tokens, min_freq=args.min_freq, max_size=args.max_vocab)
        n_stories = tokens.count(BOS)
        print(f"Рассказов: {n_stories}, слов: {len(tokens)}, словарь: {len(vocab)}. "
              f"Ячейка: {args.cell.upper()}.")

        model = StoryRNN(len(vocab), args.hidden, args.embed,
                         cell=args.cell, seed=args.seed)
        train(model, vocab, tokens, epochs=args.epochs, seq_len=args.seq_len,
              lr=args.lr, decay=args.decay)
        model.save(args.model, vocab)
        print(f"Модель сохранена: {args.model}")

    elif args.command == "story":
        model, vocab = load_or_die(args.model)
        if args.prompt:
            unknown = vocab.unknown(tokenize(args.prompt, lowercase=args.lowercase))
            if unknown:
                print(f"(слова нет в обучающем тексте, заменяю на <unk>: {unknown})",
                      file=sys.stderr)
        text = write_story(model, vocab, prompt=args.prompt,
                           sentences=args.sentences, max_words=args.max_words,
                           sentences_per_paragraph=args.paragraph,
                           min_sentence_words=args.min_sentence_words,
                           lowercase=args.lowercase,
                           rng=np.random.default_rng(args.seed),
                           **sampling_kwargs(args))
        print(text)
        if args.out:
            with open(args.out, "w", encoding="utf-8") as f:
                f.write(text + "\n")
            print(f"\n[рассказ сохранён в {args.out}]", file=sys.stderr)

    elif args.command == "chat":
        model, vocab = load_or_die(args.model)
        rng = np.random.default_rng(args.seed)
        print("Введите начало рассказа (или просто Enter — сеть придумает сама).")
        print("Для выхода наберите  выход  или нажмите Ctrl+C.\n")
        while True:
            try:
                prompt = input("Начало: ")
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if prompt.strip().lower() in {"выход", "exit", "quit"}:
                break
            text = write_story(model, vocab, prompt=prompt,
                               sentences=args.sentences, max_words=args.max_words,
                               sentences_per_paragraph=args.paragraph,
                               min_sentence_words=args.min_sentence_words,
                               lowercase=args.lowercase, rng=rng,
                               **sampling_kwargs(args))
            print("\n" + text + "\n" + "-" * 60 + "\n")
        print("Пока!")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
