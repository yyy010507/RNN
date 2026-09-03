#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
char_rnn.py — простая посимвольная рекуррентная нейронная сеть (vanilla RNN)
на чистом NumPy: без PyTorch и TensorFlow.

Что делает:
  1) учится на вашем текстовом файле (.txt) предсказывать следующий символ;
  2) в интерактивном режиме принимает слово или фразу и дописывает продолжение.

Примеры запуска
---------------
  # обучение на своём файле и сохранение модели
  python char_rnn.py train --data mytext.txt --epochs 20 --model model.npz

  # интерактивный диалог с обученной моделью
  python char_rnn.py chat --model model.npz

  # обучить и сразу перейти в диалог
  python char_rnn.py train --data mytext.txt --epochs 20 --model model.npz --chat

Зависимости: numpy.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np


# ---------------------------------------------------------------------------
# Словарь символов
# ---------------------------------------------------------------------------
class Vocab:
    """Двусторонний словарь «символ <-> индекс»."""

    def __init__(self, chars):
        self.itos = list(chars)
        self.stoi = {ch: i for i, ch in enumerate(self.itos)}

    @classmethod
    def from_text(cls, text: str) -> "Vocab":
        return cls(sorted(set(text)))

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, text: str, skip_unknown: bool = False) -> list[int]:
        if skip_unknown:
            return [self.stoi[ch] for ch in text if ch in self.stoi]
        return [self.stoi[ch] for ch in text]

    def decode(self, indices) -> str:
        return "".join(self.itos[i] for i in indices)

    def unknown(self, text: str) -> set[str]:
        return {ch for ch in text if ch not in self.stoi}


# ---------------------------------------------------------------------------
# Сама рекуррентная сеть
# ---------------------------------------------------------------------------
class CharRNN:
    """
    Классическая (vanilla) RNN с одним скрытым слоем.

    На каждом шаге t:
        h_t = tanh(W_xh @ x_t + W_hh @ h_{t-1} + b_h)
        y_t = W_hy @ h_t + b_y
        p_t = softmax(y_t)

    x_t — one-hot вектор текущего символа, p_t — распределение вероятностей
    следующего символа. Скрытое состояние h — это «память» сети о префиксе.
    """

    def __init__(self, vocab_size: int, hidden_size: int = 128, seed: int | None = None):
        rng = np.random.default_rng(seed)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        # Инициализация небольшими случайными числами (масштаб ~1/sqrt(fan_in)).
        self.Wxh = rng.standard_normal((hidden_size, vocab_size)) / np.sqrt(vocab_size)
        self.Whh = rng.standard_normal((hidden_size, hidden_size)) / np.sqrt(hidden_size)
        self.Why = rng.standard_normal((vocab_size, hidden_size)) / np.sqrt(hidden_size)
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((vocab_size, 1))

    # -- служебное ----------------------------------------------------------
    @property
    def params(self) -> dict[str, np.ndarray]:
        return {"Wxh": self.Wxh, "Whh": self.Whh, "Why": self.Why,
                "bh": self.bh, "by": self.by}

    def zero_state(self) -> np.ndarray:
        return np.zeros((self.hidden_size, 1))

    # -- прямой и обратный проход ------------------------------------------
    def loss_and_grads(self, inputs: list[int], targets: list[int], h_prev: np.ndarray):
        """
        Прямой проход по последовательности + BPTT (обратное распространение
        ошибки во времени).

        inputs  — индексы символов подаваемой последовательности;
        targets — те же символы, сдвинутые на один вперёд (что надо предсказать);
        h_prev  — скрытое состояние с предыдущего фрагмента текста.

        Возвращает: (средняя кросс-энтропия на символ, словарь градиентов,
                     последнее скрытое состояние).
        """
        xs, hs, ps = {}, {-1: h_prev.copy()}, {}
        loss = 0.0

        # --- forward ---
        for t, idx in enumerate(inputs):
            x = np.zeros((self.vocab_size, 1))
            x[idx, 0] = 1.0                              # one-hot
            xs[t] = x
            hs[t] = np.tanh(self.Wxh @ x + self.Whh @ hs[t - 1] + self.bh)
            y = self.Why @ hs[t] + self.by
            y -= y.max()                                 # стабильный softmax
            e = np.exp(y)
            ps[t] = e / e.sum()
            loss += -np.log(ps[t][targets[t], 0] + 1e-12)

        # --- backward (BPTT) ---
        grads = {k: np.zeros_like(v) for k, v in self.params.items()}
        dh_next = np.zeros_like(h_prev)

        for t in reversed(range(len(inputs))):
            dy = ps[t].copy()
            dy[targets[t], 0] -= 1.0                     # dL/dy для softmax+CE
            grads["Why"] += dy @ hs[t].T
            grads["by"] += dy

            dh = self.Why.T @ dy + dh_next
            draw = (1.0 - hs[t] ** 2) * dh               # производная tanh
            grads["bh"] += draw
            grads["Wxh"] += draw @ xs[t].T
            grads["Whh"] += draw @ hs[t - 1].T
            dh_next = self.Whh.T @ draw

        n = max(len(inputs), 1)
        for g in grads.values():
            g /= n                                       # усредняем по символам
            np.clip(g, -5.0, 5.0, out=g)                 # защита от взрыва градиентов

        return loss / n, grads, hs[len(inputs) - 1]

    # -- генерация ----------------------------------------------------------
    def _step(self, idx: int, h: np.ndarray):
        x = np.zeros((self.vocab_size, 1))
        x[idx, 0] = 1.0
        h = np.tanh(self.Wxh @ x + self.Whh @ h + self.bh)
        y = self.Why @ h + self.by
        return y, h

    def generate(self, vocab: Vocab, prompt: list[int], length: int,
                 temperature: float = 0.8, rng: np.random.Generator | None = None,
                 stop_at: str = "") -> str:
        """
        «Прогревает» сеть подсказкой prompt и дописывает length символов.

        temperature < 1 — текст более предсказуемый, > 1 — более разнообразный.
        stop_at — набор символов, на которых генерацию можно оборвать досрочно.
        """
        rng = rng or np.random.default_rng()
        h = self.zero_state()

        if not prompt:
            raise ValueError("Пустая подсказка: нечего продолжать.")

        # прогоняем подсказку, чтобы накопить контекст
        for idx in prompt[:-1]:
            _, h = self._step(idx, h)

        idx = prompt[-1]
        out = []
        for _ in range(length):
            y, h = self._step(idx, h)
            y = y / max(temperature, 1e-6)
            y -= y.max()
            e = np.exp(y)
            p = (e / e.sum()).ravel()
            idx = int(rng.choice(len(p), p=p))
            ch = vocab.itos[idx]
            out.append(ch)
            if stop_at and ch in stop_at and len(out) > 3:
                break
        return "".join(out)

    # -- сохранение / загрузка ---------------------------------------------
    def save(self, path: str, vocab: Vocab) -> None:
        np.savez_compressed(path, chars=np.array(vocab.itos, dtype=object),
                            hidden_size=self.hidden_size, **self.params)

    @classmethod
    def load(cls, path: str):
        data = np.load(path, allow_pickle=True)
        vocab = Vocab(list(data["chars"]))
        model = cls(len(vocab), int(data["hidden_size"]))
        for name in ("Wxh", "Whh", "Why", "bh", "by"):
            setattr(model, name, data[name])
        return model, vocab


# ---------------------------------------------------------------------------
# Обучение (RMSProp)
# ---------------------------------------------------------------------------
def train(model: CharRNN, vocab: Vocab, text: str, *, epochs: int = 20,
          seq_len: int = 40, lr: float = 0.002, decay: float = 0.95,
          log_every: int = 200, sample_prompt: str = "",
          verbose: bool = True) -> CharRNN:
    data = np.array(vocab.encode(text), dtype=np.int64)
    if len(data) < seq_len + 1:
        raise ValueError(f"Слишком мало текста: нужно хотя бы {seq_len + 1} символов.")

    mem = {k: np.zeros_like(v) for k, v in model.params.items()}
    smooth_loss = np.log(len(vocab))     # ожидаемая ошибка у случайной модели
    step = 0
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        h = model.zero_state()
        for p in range(0, len(data) - seq_len - 1, seq_len):
            inputs = data[p:p + seq_len].tolist()
            targets = data[p + 1:p + seq_len + 1].tolist()

            loss, grads, h = model.loss_and_grads(inputs, targets, h)
            smooth_loss = 0.999 * smooth_loss + 0.001 * loss

            # RMSProp: шаг делится на скользящее среднее модуля градиента,
            # поэтому редкие символы учатся не медленнее частых
            for name, param in model.params.items():
                g = grads[name]
                mem[name] = decay * mem[name] + (1.0 - decay) * g * g
                param -= lr * g / (np.sqrt(mem[name]) + 1e-8)

            step += 1
            if verbose and log_every and step % log_every == 0:
                msg = (f"эпоха {epoch:>3}/{epochs}  шаг {step:>6}  "
                       f"ошибка {smooth_loss:.3f}  ({time.time() - t0:.0f} с)")
                if sample_prompt:
                    prompt = vocab.encode(sample_prompt, skip_unknown=True)
                    if prompt:
                        tail = model.generate(vocab, prompt, 60, temperature=0.7)
                        msg += f"\n    проба: {sample_prompt}{tail!r}"
                print(msg, flush=True)

    if verbose:
        print(f"Обучение завершено: {step} шагов, итоговая ошибка {smooth_loss:.3f}, "
              f"{time.time() - t0:.0f} с.\n")
    return model


# ---------------------------------------------------------------------------
# Интерактивный режим
# ---------------------------------------------------------------------------
def chat(model: CharRNN, vocab: Vocab, *, length: int = 120,
         temperature: float = 0.8, seed: int | None = None) -> None:
    rng = np.random.default_rng(seed)
    print("Введите слово или фразу — сеть допишет продолжение.")
    print("Пустая строка или Ctrl+C — выход.\n")

    while True:
        try:
            prompt = input("Вы: ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not prompt.strip():
            break

        unknown = vocab.unknown(prompt)
        if unknown:
            print(f"  (символы {sorted(unknown)} не встречались в обучающем "
                  f"тексте — пропускаю их)")

        idx = vocab.encode(prompt, skip_unknown=True)
        if not idx:
            print("  Ни один символ фразы не знаком модели. Попробуйте другую.\n")
            continue

        tail = model.generate(vocab, idx, length, temperature=temperature, rng=rng)
        print(f"Сеть: {prompt}{tail}\n")

    print("Пока!")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Простая посимвольная RNN на NumPy: продолжает введённую фразу.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    tr = sub.add_parser("train", help="обучить модель на текстовом файле")
    tr.add_argument("--data", required=True, help="путь к обучающему .txt (UTF-8)")
    tr.add_argument("--model", default="model.npz", help="куда сохранить модель")
    tr.add_argument("--hidden", type=int, default=128, help="размер скрытого слоя")
    tr.add_argument("--seq-len", type=int, default=40, help="длина фрагмента для BPTT")
    tr.add_argument("--epochs", type=int, default=20, help="число проходов по тексту")
    tr.add_argument("--lr", type=float, default=0.002, help="скорость обучения")
    tr.add_argument("--decay", type=float, default=0.95,
                    help="сглаживание RMSProp (0.9-0.99)")
    tr.add_argument("--seed", type=int, default=0, help="зерно генератора случайных чисел")
    tr.add_argument("--sample-prompt", default="", help="фраза для промежуточных проб")
    tr.add_argument("--chat", action="store_true",
                    help="после обучения сразу перейти в интерактивный режим")
    tr.add_argument("--length", type=int, default=120, help="длина продолжения в чате")
    tr.add_argument("--temperature", type=float, default=0.8,
                    help="разнообразие: <1 — предсказуемее, >1 — свободнее")

    ch = sub.add_parser("chat", help="диалог с уже обученной моделью")
    ch.add_argument("--model", default="model.npz", help="путь к сохранённой модели")
    ch.add_argument("--length", type=int, default=120, help="длина продолжения")
    ch.add_argument("--temperature", type=float, default=0.8, help="разнообразие")
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
        if not text.strip():
            print("Обучающий файл пуст.", file=sys.stderr)
            return 1

        vocab = Vocab.from_text(text)
        print(f"Текст: {len(text)} символов, словарь: {len(vocab)} уникальных.")
        model = CharRNN(len(vocab), args.hidden, seed=args.seed)
        train(model, vocab, text, epochs=args.epochs, seq_len=args.seq_len,
              lr=args.lr, decay=args.decay, sample_prompt=args.sample_prompt)
        model.save(args.model, vocab)
        print(f"Модель сохранена: {args.model}")

        if args.chat:
            chat(model, vocab, length=args.length, temperature=args.temperature)

    elif args.command == "chat":
        if not os.path.exists(args.model):
            print(f"Модель не найдена: {args.model}. Сначала запустите "
                  f"`python {os.path.basename(__file__)} train --data ваш.txt`",
                  file=sys.stderr)
            return 1
        model, vocab = CharRNN.load(args.model)
        chat(model, vocab, length=args.length, temperature=args.temperature,
             seed=args.seed)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
