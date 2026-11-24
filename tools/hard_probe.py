#!/usr/bin/env python3
"""Сложная батарея: шесть задач с однозначным ответом, каждая спрашивается несколько раз.

Проверяет не «связность», а мышление: многошаговая арифметика, логика с ловушкой, алиасинг в Python,
счёт с ограничением, отслеживание состояния и точное следование формату.  Ответы проверяются машиной,
поэтому видно, где именно модель «поглупела», а не «стало хуже в целом».

    python3 tools/hard_probe.py           # 3 повтора на задачу, ~2 минуты
    N=5 python3 tools/hard_probe.py
"""
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import quality_probe as qp   # noqa: E402  - тот же запрос: greedy, thinking off, хост из конфига

N = int(os.environ.get("N") or 3)


def num(n):
    """Число как отдельное слово (27, а не 271 в «2710»)."""
    return lambda t: re.search(rf"(?<!\d){n}(?!\d)", t) is not None


TASKS = [
    # многошаговая арифметика: три неизвестных, одно уравнение
    ("arith", "В трёх корзинах 108 яблок. В первой в 2 раза больше, чем во второй, а в третьей на 12 "
              "меньше, чем в первой. Сколько яблок во второй корзине? Ответь одним числом.", num(24), 300),
    # логика: вывод не следует (обратное утверждение); «нет» в любом виде, но не «да»
    ("logic", "Все инженеры в отделе знают Python. Некоторые из тех, кто знает Python, знают Rust. "
              "Следует ли из этого, что хотя бы один инженер знает Rust? Ответь «да» или «нет».",
     lambda t: re.search(r"нет|не следует|нельзя|не обязательно", t.lower()) is not None
               and not re.match(r"\s*да\b", t.lower()), 300),
    # алиасинг: a и b — один и тот же список
    ("alias", "Что выведет этот код? Ответь ровно тем, что он печатает.\n"
              "a = [1, 2, 3]\nb = a\nb.append(4)\na[0] = 0\nprint(a, b)",
     lambda t: len(re.findall(r"0,\s*2,\s*3,\s*4", t)) >= 2, 300),
    # счёт с ограничением: кратные 3, но не кратные 5
    ("count3not5", "Сколько целых чисел от 1 до 100 делятся на 3, но не делятся на 5? Ответь одним числом.",
     num(27), 300),
    # состояние: остаток плюс столько же
    ("state", "В вазе было 5 конфет. Аня съела 2 конфеты. Потом Петя положил в вазу столько конфет, "
              "сколько в ней осталось. Сколько конфет в вазе теперь? Ответь одним числом.", num(6), 300),
    # формат: ровно пять слов на «п»
    ("format5", "Напиши ровно пять слов, каждое начинается на букву «п». Только слова, через запятую, "
                "без пояснений и без нумерации.",
     lambda t: (lambda w: len(w) == 5 and all(x.startswith("п") for x in w))(
         [x.strip().strip(".").lower() for x in t.strip().strip(".").split(",") if x.strip()]), 200),
]


def main():
    cfg = qp.cfg_load()
    fails = 0
    print(f"конфиг: {cfg_path()}   повторов на задачу: {N}")
    print("\n| задача | ответы | итог |")
    print("|---|---|---|")
    for name, prompt, check, cap in TASKS:
        seen = []
        for _ in range(N):
            text, err, finish = qp.ask(cfg, prompt, cap)
            seen.append(err or (text or "").strip().replace(chr(10), " "))
        ok_n = sum(1 for s in seen if not s.startswith(("нет соединения", "ошибка", "не JSON")) and check(s))
        if ok_n < N:
            fails += 1
        shown = " / ".join(s[:60] for s in seen[:3])
        print(f"| {name} | {shown} | {'ок' if ok_n == N else f'ПРОВАЛ {ok_n}/{N}'} |")
    print(f"\nзадач с провалом: {fails} из {len(TASKS)}")
    return fails


def cfg_path():
    return str(Path(os.environ.get("CFG") or Path(__file__).resolve().parent.parent / "strata-dsv4-ud-iq2_xxs.json"))


if __name__ == "__main__":
    sys.exit(main())
