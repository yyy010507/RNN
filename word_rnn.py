#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
word_rnn.py — простая пословная рекуррентная нейронная сеть (vanilla RNN)
на чистом NumPy: без PyTorch и TensorFlow.

Отличие от посимвольной версии: единица последовательности — слово, а не буква.
Сеть предсказывает СЛЕДУЮЩЕЕ СЛОВО, поэтому продолжение всегда состоит из
настоящих слов обучающего текста, а не из выдуманных буквосочетаний.

Что делает:
  1) учится на вашем текстовом файле (.txt) предсказывать следующее слово;
  2) в интерактивном режиме принимает слово или фразу и дописывает продолжение.

Примеры запуска
---------------
  # обучение на своём файле и сохранение модели
  python word_rnn.py train --data mytext.txt --epochs 30 --model model.npz

  # интерактивный диалог с обученной моделью
  python word_rnn.py chat --model model.npz

  # обучить и сразу перейти в диалог
  python word_rnn.py train --data mytext.txt --epochs 30 --model model.npz --chat

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


UNK = "<unk>"

# Слово = буквы/цифры/дефис-апостроф внутри слова; знак препинания = отдельный токен.
_TOKEN_RE = re.compile(r"\w+(?:[-'’]\w+)*|[^\w\s]", re.UNICODE)

# Правила «склейки» токенов обратно в текст.
_NO_SPACE_BEFORE = set(".,!?;:%)]}»…\"'’")
_NO_SPACE_AFTER = set("([{«\"'“")
_SENTENCE_END = {".", "!", "?", "…"}


def tokenize(text: str, lowercase: bool = False) -> list[str]:
    """Режет текст на слова и знаки препинания."""
    if lowercase:
        text = text.lower()
    return _TOKEN_RE.findall(text)


def detokenize(tokens) -> str:
    """Собирает токены обратно в читаемую строку."""
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
    return "".join(out)


# ---------------------------------------------------------------------------
# Словарь слов
# ---------------------------------------------------------------------------
class Vocab:
    """Двусторонний словарь «слово <-> индекс» с токеном <unk> для редких слов."""

    def __init__(self, tokens):
        self.itos = list(tokens)
        if UNK not in self.itos:
            self.itos.insert(0, UNK)
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        self.unk_id = self.stoi[UNK]

    @classmethod
    def from_tokens(cls, tokens, min_freq: int = 1, max_size: int | None = None):
        """
        Строит словарь по обучающему тексту.

        min_freq — слова, встретившиеся реже, заменяются на <unk>. Это резко
        уменьшает словарь (а с ним и размер выходного слоя) и не даёт сети
        тратить ёмкость на слова, которые она видела один раз.
        """
        counts = Counter(tokens)
        words = [w for w, c in counts.most_common() if c >= min_freq]
        if max_size is not None:
            words = words[:max_size]
        return cls(words)

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, tokens) -> list[int]:
        return [self.stoi.get(w, self.unk_id) for w in tokens]

    def decode(self, indices) -> list[str]:
        return [self.itos[i] for i in indices]

    def unknown(self, tokens) -> list[str]:
        return [w for w in tokens if w not in self.stoi]


# ---------------------------------------------------------------------------
# Сама рекуррентная сеть
# ---------------------------------------------------------------------------
class WordRNN:
    """
    Vanilla RNN с одним скрытым слоем и обучаемыми эмбеддингами слов.

    На каждом шаге t:
        e_t = E[:, word_t]                       # вектор слова (эмбеддинг)
        h_t = tanh(W_xh @ e_t + W_hh @ h_{t-1} + b_h)
        y_t = W_hy @ h_t + b_y
        p_t = softmax(y_t)                       # вероятности следующего слова

    Эмбеддинги вместо one-hot нужны потому, что словарь слов в сотни раз больше
    алфавита: плотный вектор длиной ~64 куда экономнее вектора длиной |V|.
    """

    def __init__(self, vocab_size: int, hidden_size: int = 192,
                 embed_size: int = 64, seed: int | None = None):
        rng = np.random.default_rng(seed)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size

        self.E = rng.standard_normal((embed_size, vocab_size)) * 0.01
        self.Wxh = rng.standard_normal((hidden_size, embed_size)) / np.sqrt(embed_size)
        self.Whh = rng.standard_normal((hidden_size, hidden_size)) / np.sqrt(hidden_size)
        self.Why = rng.standard_normal((vocab_size, hidden_size)) / np.sqrt(hidden_size)
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((vocab_size, 1))

    # -- служебное ----------------------------------------------------------
    @property
    def params(self) -> dict[str, np.ndarray]:
        return {"E": self.E, "Wxh": self.Wxh, "Whh": self.Whh,
                "Why": self.Why, "bh": self.bh, "by": self.by}

    def zero_state(self) -> np.ndarray:
        return np.zeros((self.hidden_size, 1))

    # -- прямой и обратный проход ------------------------------------------
    def loss_and_grads(self, inputs: list[int], targets: list[int], h_prev: np.ndarray):
        """
        Прямой проход по последовательности слов + BPTT.

        inputs  — индексы слов подаваемого фрагмента;
        targets — те же слова, сдвинутые на одно вперёд;
        h_prev  — скрытое состояние с предыдущего фрагмента.

        Возвращает: (средняя кросс-энтропия на слово, градиенты, последнее h).
        """
        es, hs, ps = {}, {-1: h_prev.copy()}, {}
        loss = 0.0

        # --- forward ---
        for t, idx in enumerate(inputs):
            es[t] = self.E[:, idx:idx + 1]                # (embed, 1)
            hs[t] = np.tanh(self.Wxh @ es[t] + self.Whh @ hs[t - 1] + self.bh)
            y = self.Why @ hs[t] + self.by
            y -= y.max()                                  # стабильный softmax
            e = np.exp(y)
            ps[t] = e / e.sum()
            loss += -np.log(ps[t][targets[t], 0] + 1e-12)

        # --- backward (BPTT) ---
        grads = {k: np.zeros_like(v) for k, v in self.params.items()}
        dh_next = np.zeros_like(h_prev)

        for t in reversed(range(len(inputs))):
            dy = ps[t].copy()
            dy[targets[t], 0] -= 1.0                      # dL/dy для softmax+CE
            grads["Why"] += dy @ hs[t].T
            grads["by"] += dy

            dh = self.Why.T @ dy + dh_next
            draw = (1.0 - hs[t] ** 2) * dh                # производная tanh
            grads["bh"] += draw
            grads["Wxh"] += draw @ es[t].T
            grads["Whh"] += draw @ hs[t - 1].T
            # градиент уходит только в столбец использованного слова
            grads["E"][:, inputs[t]] += (self.Wxh.T @ draw).ravel()
            dh_next = self.Whh.T @ draw

        n = max(len(inputs), 1)
        for g in grads.values():
            g /= n                                        # усредняем по словам
            np.clip(g, -5.0, 5.0, out=g)                  # защита от взрыва градиентов

        return loss / n, grads, hs[len(inputs) - 1]

    # -- генерация ----------------------------------------------------------
    def _step(self, idx: int, h: np.ndarray):
        e = self.E[:, idx:idx + 1]
        h = np.tanh(self.Wxh @ e + self.Whh @ h + self.bh)
        return self.Why @ h + self.by, h

    def generate(self, vocab: Vocab, prompt: list[int], n_words: int,
                 temperature: float = 0.8, top_k: int = 0,
                 rng: np.random.Generator | None = None,
                 stop_at_sentence: bool = False) -> list[str]:
        """
        «Прогревает» сеть подсказкой и дописывает n_words слов.

        temperature < 1 — выбор увереннее, > 1 — разнообразнее.
        top_k > 0 — выбирать только из k самых вероятных слов (меньше мусора).
        """
        rng = rng or np.random.default_rng()
        if not prompt:
            raise ValueError("Пустая подсказка: нечего продолжать.")

        h = self.zero_state()
        for idx in prompt[:-1]:
            _, h = self._step(idx, h)

        idx = prompt[-1]
        out = []
        for _ in range(n_words):
            y, h = self._step(idx, h)
            y = y.ravel() / max(temperature, 1e-6)
            y[vocab.unk_id] = -np.inf                     # <unk> не выводим

            if top_k and top_k < len(y):
                keep = np.argpartition(y, -top_k)[-top_k:]
                mask = np.full_like(y, -np.inf)
                mask[keep] = y[keep]
                y = mask

            y -= y.max()
            p = np.exp(y)
            p /= p.sum()
            idx = int(rng.choice(len(p), p=p))
            word = vocab.itos[idx]
            out.append(word)
            if stop_at_sentence and word in _SENTENCE_END:
                break
        return out

    # -- сохранение / загрузка ---------------------------------------------
    def save(self, path: str, vocab: Vocab) -> None:
        np.savez_compressed(path, words=np.array(vocab.itos, dtype=object),
                            hidden_size=self.hidden_size,
                            embed_size=self.embed_size, **self.params)

    @classmethod
    def load(cls, path: str):
        data = np.load(path, allow_pickle=True)
        vocab = Vocab(list(data["words"]))
        model = cls(len(vocab), int(data["hidden_size"]), int(data["embed_size"]))
        for name in ("E", "Wxh", "Whh", "Why", "bh", "by"):
            setattr(model, name, data[name])
        return model, vocab


# ---------------------------------------------------------------------------
# Обучение (RMSProp)
# ---------------------------------------------------------------------------
def train(model: WordRNN, vocab: Vocab, tokens, *, epochs: int = 30,
          seq_len: int = 20, lr: float = 0.002, decay: float = 0.95,
          log_every: int = 200, sample_prompt: str = "",
          verbose: bool = True) -> WordRNN:
    data = np.array(vocab.encode(tokens), dtype=np.int64)
    if len(data) < seq_len + 1:
        raise ValueError(f"Слишком мало текста: нужно хотя бы {seq_len + 1} слов.")

    mem = {k: np.zeros_like(v) for k, v in model.params.items()}
    smooth_loss = np.log(len(vocab))     # ошибка модели, которая просто угадывает
    step = 0
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        h = model.zero_state()
        for p in range(0, len(data) - seq_len - 1, seq_len):
            inputs = data[p:p + seq_len].tolist()
            targets = data[p + 1:p + seq_len + 1].tolist()

            loss, grads, h = model.loss_and_grads(inputs, targets, h)
            smooth_loss = 0.999 * smooth_loss + 0.001 * loss

            # RMSProp: шаг делится на скользящее среднее модуля градиента
            for name, param in model.params.items():
                g = grads[name]
                mem[name] = decay * mem[name] + (1.0 - decay) * g * g
                param -= lr * g / (np.sqrt(mem[name]) + 1e-8)

            step += 1
            if verbose and log_every and step % log_every == 0:
                ppl = float(np.exp(min(smooth_loss, 20)))
                msg = (f"эпоха {epoch:>3}/{epochs}  шаг {step:>6}  "
                       f"ошибка {smooth_loss:.3f}  перплексия {ppl:7.1f}  "
                       f"({time.time() - t0:.0f} с)")
                if sample_prompt:
                    prompt = vocab.encode(tokenize(sample_prompt))
                    tail = model.generate(vocab, prompt, 12, temperature=0.7, top_k=10)
                    msg += "\n    проба: " + detokenize(tokenize(sample_prompt) + tail)
                print(msg, flush=True)

    if verbose:
        print(f"Обучение завершено: {step} шагов, итоговая ошибка {smooth_loss:.3f}, "
              f"{time.time() - t0:.0f} с.\n")
    return model


# ---------------------------------------------------------------------------
# Интерактивный режим
# ---------------------------------------------------------------------------
def chat(model: WordRNN, vocab: Vocab, *, n_words: int = 25,
         temperature: float = 0.8, top_k: int = 10, lowercase: bool = False,
         stop_at_sentence: bool = False, seed: int | None = None) -> None:
    rng = np.random.default_rng(seed)
    print("Введите слово или фразу — сеть допишет продолжение словами.")
    print("Пустая строка или Ctrl+C — выход.\n")

    while True:
        try:
            raw = input("Вы: ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not raw.strip():
            break

        tokens = tokenize(raw, lowercase=lowercase)
        if not tokens:
            print("  Не нашёл ни одного слова. Попробуйте другую фразу.\n")
            continue

        unknown = vocab.unknown(tokens)
        if unknown:
            print(f"  (слов{'а' if len(unknown) > 1 else 'о'} {unknown} нет в "
                  f"обучающем тексте — заменяю на <unk>)")

        prompt = vocab.encode(tokens)
        tail = model.generate(vocab, prompt, n_words, temperature=temperature,
                              top_k=top_k, rng=rng,
                              stop_at_sentence=stop_at_sentence)
        print("Сеть: " + detokenize(tokens + tail) + "\n")

    print("Пока!")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Пословная RNN на NumPy: продолжает введённую фразу словами.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    tr = sub.add_parser("train", help="обучить модель на текстовом файле")
    tr.add_argument("--data", required=True, help="путь к обучающему .txt (UTF-8)")
    tr.add_argument("--model", default="model.npz", help="куда сохранить модель")
    tr.add_argument("--hidden", type=int, default=192, help="размер скрытого слоя")
    tr.add_argument("--embed", type=int, default=64, help="размер эмбеддинга слова")
    tr.add_argument("--seq-len", type=int, default=20, help="длина фрагмента для BPTT (в словах)")
    tr.add_argument("--epochs", type=int, default=30, help="число проходов по тексту")
    tr.add_argument("--lr", type=float, default=0.002, help="скорость обучения")
    tr.add_argument("--decay", type=float, default=0.95, help="сглаживание RMSProp (0.9-0.99)")
    tr.add_argument("--min-freq", type=int, default=1,
                    help="слова реже этой частоты заменяются на <unk>")
    tr.add_argument("--max-vocab", type=int, default=None, help="ограничить размер словаря")
    tr.add_argument("--lowercase", action="store_true", help="привести текст к нижнему регистру")
    tr.add_argument("--seed", type=int, default=0, help="зерно генератора случайных чисел")
    tr.add_argument("--sample-prompt", default="", help="фраза для промежуточных проб")
    tr.add_argument("--chat", action="store_true",
                    help="после обучения сразу перейти в интерактивный режим")
    tr.add_argument("--words", type=int, default=25, help="сколько слов дописывать в чате")
    tr.add_argument("--temperature", type=float, default=0.8, help="разнообразие")
    tr.add_argument("--top-k", type=int, default=10, help="выбирать из k лучших слов (0 — из всех)")

    ch = sub.add_parser("chat", help="диалог с уже обученной моделью")
    ch.add_argument("--model", default="model.npz", help="путь к сохранённой модели")
    ch.add_argument("--words", type=int, default=25, help="сколько слов дописывать")
    ch.add_argument("--temperature", type=float, default=0.8, help="разнообразие")
    ch.add_argument("--top-k", type=int, default=10, help="выбирать из k лучших слов (0 — из всех)")
    ch.add_argument("--lowercase", action="store_true",
                    help="если модель обучена с --lowercase")
    ch.add_argument("--stop-at-sentence", action="store_true",
                    help="останавливаться в конце предложения")
    ch.add_argument("--seed", type=int, default=None, help="зерно для воспроизводимости")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "train":
        if not os.path.exists(args.data):
            print(f"Файл не найден: {args.data}", file=sys.stderr)
            return 1
        with open(args.data, encoding="utf-8") as f:
            text = f.read()

        tokens = tokenize(text, lowercase=args.lowercase)
        if not tokens:
            print("В обучающем файле нет слов.", file=sys.stderr)
            return 1

        vocab = Vocab.from_tokens(tokens, min_freq=args.min_freq,
                                  max_size=args.max_vocab)
        coverage = 100.0 * sum(w in vocab.stoi for w in tokens) / len(tokens)
        print(f"Текст: {len(tokens)} слов, словарь: {len(vocab)} "
              f"(покрытие {coverage:.1f}%).")

        model = WordRNN(len(vocab), args.hidden, args.embed, seed=args.seed)
        train(model, vocab, tokens, epochs=args.epochs, seq_len=args.seq_len,
              lr=args.lr, decay=args.decay, sample_prompt=args.sample_prompt)
        model.save(args.model, vocab)
        print(f"Модель сохранена: {args.model}")

        if args.chat:
            chat(model, vocab, n_words=args.words, temperature=args.temperature,
                 top_k=args.top_k, lowercase=args.lowercase)

    elif args.command == "chat":
        if not os.path.exists(args.model):
            print(f"Модель не найдена: {args.model}. Сначала запустите "
                  f"`python {os.path.basename(__file__)} train --data ваш.txt`",
                  file=sys.stderr)
            return 1
        model, vocab = WordRNN.load(args.model)
        chat(model, vocab, n_words=args.words, temperature=args.temperature,
             top_k=args.top_k, lowercase=args.lowercase,
             stop_at_sentence=args.stop_at_sentence, seed=args.seed)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
