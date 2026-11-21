#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""check_strings.py - ловит форматные строки, разорванные переводом строки.

    python3 tools/check_strings.py [src]

Зачем: правка файлов через shell-heredoc съедает обратные слэши, и `"...\\n"` в коде
превращается в реальный перевод строки ВНУТРИ строкового литерала - компилятор
говорит `missing closing quote` за три экрана от места, а `tail -5` сборки этого
не показывает (так уже дважды падала сборка на eagle3.cu).  Скрипт ищет строки,
где у fprintf/snprintf/printf непарное число кавычек, и завершается кодом 1.
"""
import glob
import os
import sys


def main() -> int:
    root = sys.argv[1] if len(sys.argv) > 1 else "src"
    patterns = ("/**/*.cpp", "/**/*.cu", "/**/*.cuh", "/**/*.hpp", "/**/*.h")
    files = []
    for pat in patterns:
        files += glob.glob(root + pat, recursive=True)
    bad = []
    for path in sorted(set(files)):
        with open(path, encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f.read().split("\n"), 1):
                if ("fprintf" in line or "snprintf" in line or "printf" in line) and line.count('"') % 2 == 1:
                    bad.append(f"{path}:{i}: {line.strip()[:100]}")
    if bad:
        print("разорванные форматные строки (непарные кавычки):", file=sys.stderr)
        for b in bad:
            print("  " + b, file=sys.stderr)
        print("\nпочинка: в C-строке нужен ДВА символа - обратный слэш и n; при правке через\n"
              "heredoc стройте его как chr(92) + 'n' или правьте файл редактором.", file=sys.stderr)
        return 1
    print(f"ок: {len(set(files))} файлов, разорванных строк нет")
    return 0


if __name__ == "__main__":
    sys.exit(main())
