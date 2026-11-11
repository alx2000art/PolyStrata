#!/usr/bin/env python3
# make_prompt.py - builds the reading-test prompts for knob_ab.sh: three ~30k-token files under
# ~/prompts (story / doc / code), deterministic (the same run of the script always writes the same
# bytes, so an A/B's two values always read the same prompt), Russian, varied enough that a long
# prompt routes across a wide set of experts.
#
#   python3 make_prompt.py            # writes ~/prompts/{story,doc,code}-30k.txt
#   python3 make_prompt.py /some/dir  # or any directory
#
# The sizes target ~85-95k characters, which is ~28-32k tokens of Russian for DSV4's tokenizer.
import os
import random
import sys

TARGET_CHARS = 88_000

NAMES = ["Марк", "Элина", "Горан", "Ясна", "Тейо", "Ирма", "Ловран", "Ния", "Дарен", "Софь",
         "Милан", "Агата", "Родис", "Кера", "Виктор", "Лана", "Освальд", "Мира", "Пётр", "Зарема"]
TOWNS = ["Вёрный Ключ", "Сторожевск", "Белогорье", "Тихая Протва", "Каменка", "Заречье", "Луговое",
         "Высокое", "Приморск", "Лесной Дол"]
THINGS = ["медный компас", "старый журнал", "ключ с биркой", "карта без края", "чайник со свистком",
          "фонарь на керосине", "шкатулка с двойным дном", "список имён", "фотография моря",
          "письмо без адреса", "граммофон", "часы, отстающие на семь минут"]
MOODS = ["тихо", "ветрено", "промозгло", "светло", "сумеречно", "морозно", "душно", "звонко"]
VERBS = ["заметил", "вспомнил", "переспросил", "записал", "усмехнулся", "помолчал", "подвинул",
         "проверил", "не поверил", "согласился", "нашёл", "потерял", "обещал", "постучал"]
OBJECTS = ["о дороге", "о письме", "о сроке", "о деньгах", "о погоде", "о прошлом годе",
           "о соседях", "о мосте", "о шкафе", "о назначении", "о письме без адреса"]
CONTINUE = ["И тогда", "Впрочем", "К полудню", "К вечеру", "Наутро", "Через неделю", "Тем временем",
            "Вопреки ожиданию", "Как ни странно", "Ещё до рассвета"]

def sentence(r):
    pick = r.random()
    if pick < 0.30:
        return f"{r.choice(CONTINUE)} {r.choice(NAMES)} {r.choice(VERBS)}, что разговор {r.choice(OBJECTS)} стоит отложить до {r.choice(['завтра', 'вечера', 'весны', 'пятницы', 'возвращения почтальона'])}."
    if pick < 0.55:
        return f"В {r.choice(TOWNS)} было {r.choice(MOODS)}, и {r.choice(THINGS)} лежал(а) на подоконнике, будто его кто-то оставил(а) нарочно."
    if pick < 0.75:
        a, b = r.sample(NAMES, 2)
        return f"«{r.choice(VERBS).capitalize()}?», спросил(а) {a}, и {b} ответил(а), что {r.choice(['да', 'нет', 'не знает', 'расскажет позже', 'покажет утром'])}, потому что подробности {r.choice(OBJECTS)} никого не касались."
    return f"Дорога от {r.choice(TOWNS)} до {r.choice(TOWNS)} занимала {r.choice(['час', 'полдня', 'два дня', 'всю ночь'])}, и всё это время {r.choice(NAMES)} думал(а) {r.choice(OBJECTS)}."

def chapter(r, i):
    head = f"Глава {i}. {r.choice(['Письмо', 'Мост', 'Компас', 'Погода', 'Список', 'Часы', 'Дорога', 'Шкатулка', 'Гость', 'Слух'])} {i}"
    paras = []
    for _ in range(r.randint(6, 9)):
        n = r.randint(4, 7)
        paras.append("  " + " ".join(sentence(r) for _ in range(n)))
    if r.random() < 0.7:
        a, b = r.sample(NAMES, 2)
        paras.append(f"  «{r.choice(['Мы опоздали', 'Всё записано', 'Она права', 'Он лгал', 'Завтра решится', 'Дай мне знак'])},», сказал(а) {a}. «{r.choice(['Тогда идём', 'Тогда ждём', 'Тогда молчим', 'Тогда проверим', 'Тогда уходим'])},», ответил(а) {b}.")
    return head + "\n\n" + "\n\n".join(paras)

TERMS = ["буфер чтения", "очередь заданий", "таблица страниц", "пул слотов", "поток ввода",
         "отложенная запись", "счётчик обращений", "политика вытеснения", "сегмент карты",
         "закреплённая память", "граница куска", "флаг готовности"]
TOPICS = ["выравнивание", "производительность", "надёжность", "совместимость", "диагностика",
          "безопасность", "конфигурация", "наблюдаемость", "восстановление", "распределение"]

def doc_section(r, i):
    t = r.choice(TOPICS)
    head = f"{i}. {t.capitalize()} {r.choice(TERMS)}"
    body = [f"  В этом разделе описывается {t} {r.choice(TERMS)}: когда оно применяется, какие ограничения имеет и как проверяется. Значения по умолчанию выбраны консервативно; исключения перечислены в пунктах ниже."]
    for k in range(r.randint(4, 7)):
        body.append(f"  {i}.{k+1}. {r.choice(['Если', 'Когда', 'При условии, что', 'В случае, если'])} {r.choice(TERMS)} занимает больше {r.choice(['256', '512', '1024', '2048'])} МиБ, рекомендуется {r.choice(['уменьшить', 'зафиксировать', 'пересчитать', 'ограничить'])} его размер и {r.choice(['проверить журнал', 'измерить задержку', 'сверить счётчики', 'повторить прогон'])}.")
    body.append(f"  Пример: {r.choice(TERMS)} размера {r.choice(['512', '1024'])} МиБ при {r.choice(['32', '64', '128'])} потоках даёт задержку {r.choice(['0.4', '1.2', '3.5', '7.0'])} мс; при вдвое большем буфере задержка растёт {r.choice(['незначительно', 'примерно вдвое', 'нелинейно'])}.")
    body.append(f"  Замечание: {r.choice(TOPICS)} {r.choice(TERMS)} не заменяет {r.choice(TERMS)}; они настраиваются независимо и проверяются по отдельности.")
    return head + "\n\n" + "\n".join(body)

CODE_SNIPPETS = [
    ("resize_pool", ["size_t need = slots * slot_bytes;", "if (need > budget) slots = budget / slot_bytes;", "return pool_resize(slots);"]),
    ("count_hits", ["for (int i = 0; i < n; ++i) hits += table[ids[i]] >= 0;", "return hits;"]),
    ("evict_lru", ["int victim = -1; uint64_t best = ~0ull;", "for (int s = 0; s < n_slots; ++s) if (!busy[s] && tick[s] < best) best = tick[s], victim = s;", "return victim;"]),
    ("stage_copy", ["memcpy(dst, src, bytes);", "flush(dst);", "mark_ready(chunk);"]),
]

def code_block(r, i):
    name, lines = r.choice(CODE_SNIPPETS)
    head = f"## Пример {i}: {name.replace('_', ' ')}"
    body = f"```python\ndef {name}(n_slots, slot_bytes, budget, table, ids, n):\n"
    for ln in lines:
        body += f"    {ln}\n"
    body += "```\n"
    note = f"Функция {name} вызывается {r.choice(['на каждом шаге', 'раз в окно', 'при переполнении', 'по требованию оператора'])}; ошибки не поднимаются, а записываются в журнал и сбрасываются счётчиком."
    return head + "\n\n" + body + note + "\n"

def fill(make, r, title):
    out = [title + "\n"]
    total = 0
    i = 0
    while total < TARGET_CHARS:
        piece = make(r, i + 1)
        out.append(piece)
        total += len(piece)
        i += 1
    return "\n\n".join(out)

def main():
    dest = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/prompts")
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 1234   # a new seed: fresh text for a cold test
    os.makedirs(dest, exist_ok=True)
    for name, maker, title in (
        ("story-30k", lambda r, i: chapter(r, i), "Рассказ: Регистр и медный компас"),
        ("doc-30k", lambda r, i: doc_section(r, i), "Техническое описание: буферы, пулы и их настройка"),
        ("code-30k", lambda r, i: code_block(r, i), "Рабочая тетрадь: примеры и пояснения"),
    ):
        text = fill(maker, random.Random(seed + hash(name) % 1000), title)
        path = os.path.join(dest, name + ".txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"{path}: {len(text):,} chars (~{len(text) // 2900}k tokens)")
    return 0

if __name__ == "__main__":
    sys.exit(main())
