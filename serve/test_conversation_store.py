import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

from serve.conversation_store import ConversationStore, MIN_GROW, MIN_TOKENS


class FakeEngine:
    """session_file как у StrataEngine: SAVE пишет файл, RESTORE его читает; причины отказа — ValueError."""

    def __init__(self, size=5):
        self.size = size
        self.saved = []            # (path, tokens)
        self.restored = []         # paths
        self.refuse_restore = None

    def session_file(self, action, path):
        if action == "save":
            tokens = 5000
            Path(path).write_bytes(b"x" * self.size)
            self.saved.append((path, tokens))
            return {"tokens": tokens, "bytes": self.size, "ms": 1.0}
        if self.refuse_restore:
            raise ValueError(self.refuse_restore)
        if not Path(path).is_file():
            raise OSError("no such file")
        self.restored.append(path)
        return {"tokens": 5000, "bytes": 5, "ms": 1.0}


class Store(unittest.TestCase):
    def store(self, d, slots=5):
        return ConversationStore(d, slots=slots)

    def test_key_from_the_head(self):
        s = self.store(tempfile.mkdtemp())
        head = list(range(1, 2001))
        self.assertEqual(s.key(head), s.key(head + [999999]))       # продолжение того же чата
        self.assertNotEqual(s.key(head), s.key([7] + head[1:]))     # первый токен другой
        self.assertEqual(len(s.key(head)), 16)

    def test_short_chats_and_growth(self):
        s = self.store(tempfile.mkdtemp())
        self.assertFalse(s.worth_saving("k", MIN_TOKENS - 1))
        self.assertTrue(s.worth_saving("k", MIN_TOKENS))
        s.saved["k"] = MIN_TOKENS
        self.assertFalse(s.worth_saving("k", MIN_TOKENS + MIN_GROW - 1))   # почти не вырос
        self.assertTrue(s.worth_saving("k", MIN_TOKENS + MIN_GROW))

    def test_save_replaces_atomically_and_leaves_no_tmp(self):
        d = tempfile.mkdtemp()
        s, e = self.store(d), FakeEngine()
        s.save(e, "abc", MIN_TOKENS)
        self.assertTrue(s.file("abc").is_file())
        self.assertEqual(sorted(p.name for p in Path(d).iterdir()), ["conv-abc.spcs"])
        self.assertEqual(s.saved["abc"], 5000)

    def test_switch_saves_the_live_chat_and_restores_a_known_one(self):
        s, e = self.store(tempfile.mkdtemp()), FakeEngine()
        a, b = [1] * 3000, [2] * 3000
        s.switch(e, a)                                   # первый чат: файла нет, только запомнили
        self.assertEqual(e.saved, [])
        s.touch(len(a) + 200)
        s.switch(e, a)                                   # тот же чат: ничего не пишется и не читается
        self.assertEqual((e.saved, e.restored), ([], []))
        s.switch(e, b)                                   # другой чат: A ушёл в файл
        self.assertEqual(len(e.saved), 1)
        self.assertFalse(s.file(s.key(b)).exists())
        s.switch(e, a)                                   # возврат: A поднят из файла (B сохранён)
        self.assertEqual(e.restored, [str(s.file(s.key(a)))])

    def test_short_live_chat_is_not_saved(self):
        s, e = self.store(tempfile.mkdtemp()), FakeEngine()
        s.switch(e, [1] * 100)
        s.switch(e, [2] * 100)
        self.assertEqual(e.saved, [])

    def test_another_models_file_is_left_for_that_model(self):
        d = tempfile.mkdtemp()
        s, e = self.store(d), FakeEngine()
        a = [1] * 3000
        s.switch(e, a)
        s.switch(e, [2] * 3000)                          # A в файле
        e.refuse_restore = "another model or build"
        s.switch(e, a)                                   # отказ: запрос продолжается, файл не тронут
        self.assertTrue(s.file(s.key(a)).exists())

    def test_eviction_keeps_the_newest(self):
        d = tempfile.mkdtemp()
        s, e = self.store(d, slots=2), FakeEngine()
        for key in ("aa", "bb", "cc"):
            s.save(e, key, MIN_TOKENS)
            time.sleep(0.01)
        self.assertEqual(sorted(p.name for p in Path(d).glob("conv-*.spcs")), ["conv-bb.spcs", "conv-cc.spcs"])

    def test_eviction_never_drops_the_one_just_written(self):
        d = tempfile.mkdtemp()
        s, e = self.store(d, slots=1), FakeEngine()
        s.save(e, "aa", MIN_TOKENS)
        time.sleep(0.02)
        for p in Path(d).glob("conv-*.spcs"):
            p.touch()                                    # aa стало новее только что записанного
        s.save(e, "bb", MIN_TOKENS)
        self.assertTrue(s.file("bb").exists())

    def test_byte_budget_drops_the_oldest(self):
        d = tempfile.mkdtemp()
        s, e = self.store(d), FakeEngine(size=1000)
        s.max_bytes = 2500
        for key in ("aa", "bb", "cc"):
            s.save(e, key, MIN_TOKENS)
            time.sleep(0.01)
        # три файла по 1000 байт при бюджете 2500: уходит самый старый, остаются два свежих
        self.assertEqual(sorted(p.name for p in Path(d).glob("conv-*.spcs")),
                         ["conv-bb.spcs", "conv-cc.spcs"])

    def test_save_live_only_when_it_grew(self):
        d = tempfile.mkdtemp()
        s, e = self.store(d), FakeEngine()               # the fake's SAVE answers 5000 tokens
        self.assertIsNone(s.save_live(e))                 # nothing live yet
        s.switch(e, [1] * 3000)
        self.assertIsNotNone(s.save_live(e))              # idle: the chat is worth its first file
        self.assertTrue(s.file(s.key([1] * 3000)).exists())
        self.assertIsNone(s.save_live(e))                 # nothing new since: no write
        s.touch(5000 + MIN_GROW)                          # it grew by the floor since the file
        self.assertIsNotNone(s.save_live(e))

    def test_one_file_bigger_than_the_budget_stays(self):
        d = tempfile.mkdtemp()
        s, e = self.store(d), FakeEngine(size=5000)
        s.max_bytes = 100                                 # меньше одного файла
        s.save(e, "aa", MIN_TOKENS)
        s.save(e, "bb", MIN_TOKENS)
        self.assertTrue(s.file("bb").exists())            # только что записанный не трогается
        self.assertFalse(s.file("aa").exists())           # а прежний под бюджет не подходит
        self.assertEqual(s.bytes_held(), 5000)


if __name__ == "__main__":
    unittest.main()
