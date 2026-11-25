"""Дисковый кэш разговоров (opt-in, --conversation-store DIR): разговор, ушедший из движка, ложится в файл,
а вернувшийся — поднимается из него, вместо того чтобы читать промпт заново.

Разговор узнаётся по первым KEY_TOKENS (1024) своим токенам: у продолжения того же чата они те же (шаблон
рендерит историю одинаково), у другого чата — другие.  Короче MIN_TOKENS (2048) не сохраняется: у такого
чтение дешевле записи, и первые 1024 токена ещё не устоялись (первый ход короткий — ключ сменился бы на
втором).  Файлы — conv-<ключ>.spcs, не больше slots штук и max_mib мегабайт; самые давно использованные
удаляются (один файл, только что записанный, может выйти за бюджет: он не трогается).

Сам файл пишет движок (SAVE <path> / RESTORE <path>, session_file у сервера): здесь только имена, замена
через временный файл (оборванная запись не портит прежний), учёт роста и уборка старых.  Ошибки хранилища
не валят запрос: пишутся в log, разговор просто прочитается заново (исключение — движок умер: это решает
сервер, не хранилище).
"""
import hashlib
import os
import time
from pathlib import Path

KEY_TOKENS = 1024      # первые токены, по которым разговор узнаётся
MIN_TOKENS = 2048      # короче этого файл не пишется: восстановление не окупает запись
MIN_GROW = 2048        # столько токенов должно прибавиться, чтобы сохранять разговор ещё раз
MAX_MIB = 10240        # бюджет хранилища, когда не задан другой


class ConversationStore:
    DEFAULT_MAX_MIB = MAX_MIB

    def __init__(self, path, slots=5, max_mib=MAX_MIB, log=None):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.slots = max(1, int(slots))
        self.max_bytes = max(0, int(max_mib)) << 20   # 0: только по числу файлов
        self.log = log or (lambda message: None)
        self.live = None       # (key, tokens): разговор, который держит движок, и сколько в нём токенов
        self.saved = {}        # key -> токенов в файле, который мы записали в этом запуске

    @staticmethod
    def key(ids):
        """Имя разговора по его первым токенам."""
        head = ",".join(str(int(t)) for t in ids[:KEY_TOKENS])
        return hashlib.sha1(head.encode()).hexdigest()[:16]

    def file(self, key):
        return self.path / f"conv-{key}.spcs"

    def has(self, key):
        return self.file(key).is_file()

    def worth_saving(self, key, tokens):
        """Есть ли смысл писать файл сейчас: разговор достаточно длинный, а с прошлой записи заметно вырос."""
        if tokens < MIN_TOKENS:
            return False
        was = self.saved.get(key)
        return was is None or tokens - was >= MIN_GROW

    def switch(self, engine, ids):
        """Перед запросом: прежний разговор уходит в файл, этот поднимается из своего, если файл есть.
        Вызывается между запросами, когда движок держит прежний разговор."""
        key = self.key(ids)
        live = self.live
        if live is not None and live[0] != key:
            self._save(engine, live[0], live[1])
        if (live is None or live[0] != key) and self.has(key):
            self._restore(engine, key)
        self.live = (key, len(ids))
        return key

    def touch(self, tokens):
        """Разговор, который держит движок, вырос: столько в нём теперь токенов."""
        if self.live is not None:
            self.live = (self.live[0], tokens)

    def save(self, engine, key, tokens):
        """Движок кладёт разговор в файл; временный файл заменяет прежний только после успеха."""
        final = self.file(key)
        tmp = final.with_name(f".{final.name}.{os.getpid()}.tmp")
        try:
            r = engine.session_file("save", str(tmp))
            os.replace(tmp, final)
            os.utime(final)
        finally:
            try:
                tmp.unlink()
            except OSError:
                pass
        self.saved[key] = r["tokens"]
        self.evict(keep=key)
        return r

    def restore(self, engine, key):
        r = engine.session_file("restore", str(self.file(key)))
        self.saved[key] = r["tokens"]   # в файле столько: возврат без нового хода не стоит писать заново
        try:
            os.utime(self.file(key))    # самое свежее использование: уборка идёт по mtime
        except OSError:
            pass
        return r

    def drop(self, key):
        """Файл, который движок отказался принять (другая модель или сборка), больше не нужен."""
        try:
            self.file(key).unlink()
        except OSError:
            pass
        self.saved.pop(key, None)

    def evict(self, keep=None):
        """Оставить не больше slots файлов и max_bytes байт: самые давно использованные удаляются.  `keep` -
        только что записанный или прочитанный - не трогается, даже когда он один больше бюджета."""
        files = []
        for p in self.path.glob("conv-*.spcs"):
            try:
                st = p.stat()
            except OSError:
                continue
            files.append((st.st_mtime, st.st_size, p))
        files.sort(key=lambda item: item[0], reverse=True)     # свежие первыми
        total = sum(size for _, size, _ in files)
        drop = [p for i, (_, _, p) in enumerate(files) if i >= self.slots]
        for i, (_, size, _) in enumerate(files):
            if i >= self.slots:
                total -= size
        for _, size, p in reversed(files[:self.slots]):        # от старых к свежим
            if self.max_bytes == 0 or total <= self.max_bytes:
                break
            if keep is not None and p.name == self.file(keep).name:
                continue
            drop.append(p)
            total -= size
        for p in drop:
            if keep is not None and p.name == self.file(keep).name:
                continue
            try:
                p.unlink()
            except OSError:
                continue
            self.saved.pop(p.stem[len("conv-"):], None)

    def bytes_held(self):
        """Сколько занимают файлы хранилища сейчас."""
        total = 0
        for p in self.path.glob("conv-*.spcs"):
            try:
                total += p.stat().st_size
            except OSError:
                pass
        return total

    # ---- the two calls the server makes, with their failures kept out of the request ----
    @staticmethod
    def _why(error):
        kind = getattr(error, "kind", None)
        return f"{kind}: {error}" if kind else str(error)

    def _save(self, engine, key, tokens):
        if not self.worth_saving(key, tokens):
            return None
        t0 = time.time()
        try:
            r = self.save(engine, key, tokens)
        except (ValueError, OSError) as e:
            self.log(f"conversation store: the chat was not saved ({self._why(e)})")
            return None
        self.log(f"conversation store: {r['tokens']} tokens, {r['bytes'] / 2**20:.1f} MiB -> "
                 f"{self.file(key).name} in {time.time() - t0:.1f} s")
        return r

    def _restore(self, engine, key):
        t0 = time.time()
        try:
            r = self.restore(engine, key)
        except ValueError as e:
            # чужой файл (другая модель или сборка) не трогаем: к той модели его ещё вернут; чтение отказа
            # стоит копейки - движок проверяет заголовок и не берёт ничего из файла
            self.log(f"conversation store: {self.file(key).name} cannot be used ({self._why(e)}); left as it is")
            return None
        except OSError as e:
            self.log(f"conversation store: {self.file(key).name} cannot be read ({e}); left for later")
            return None
        self.log(f"conversation store: {r['tokens']} tokens, {r['bytes'] / 2**20:.1f} MiB back from "
                 f"{self.file(key).name} in {time.time() - t0:.1f} s")
        return r
