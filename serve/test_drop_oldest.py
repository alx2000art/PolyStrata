"""#17: "drop_oldest" leaves out a conversation's oldest turns when it no longer fits the context.

    python -m unittest serve.test_drop_oldest -v
"""
import unittest
from pathlib import Path

from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, MockEngine, Service

ROOT = Path(__file__).resolve().parents[1]
CTX = 4096   # bytes are tokens here: a turn of 1,000 characters is about 1,000 tokens


def turn(i):
    return [{"role": "user", "content": f"question {i} " + "q" * 1000},
            {"role": "assistant", "content": f"answer {i} " + "a" * 1000}]


class DropOldest(unittest.TestCase):
    def setUp(self):
        tok = ByteTokenizer()
        self.tok = tok
        self.svc = Service(MockEngine(tok, "ok", max_context=CTX), tok,
                           ChatTemplate(ROOT / "serve/chat_template.jinja"))

    def prompt(self, messages, max_new=None):
        ids, _, max_new = self.svc.prepare(messages, None, {}, max_new)
        return self.tok.decode([t for t in ids if t < 256]), max_new

    def test_off_refuses_as_before(self):
        messages = [m for i in range(4) for m in turn(i)] + [{"role": "user", "content": "last"}]
        with self.assertRaisesRegex(ValueError, "no room to answer"):
            self.svc.prepare(messages, None, {}, None)

    def test_on_keeps_the_system_prompt_and_the_newest_turns(self):
        self.svc.drop_oldest = True
        messages = ([{"role": "system", "content": "the system prompt"}] + [m for i in range(4) for m in turn(i)] +
                    [{"role": "user", "content": "the last question"}])
        text, max_new = self.prompt(messages)
        self.assertIn("the system prompt", text)
        self.assertIn("the last question", text)
        self.assertNotIn("question 0", text)
        self.assertNotIn("answer 0", text)
        self.assertIn("answer 3", text)
        self.assertGreaterEqual(max_new, CTX // 8)

    def test_a_short_conversation_is_untouched(self):
        self.svc.drop_oldest = True
        messages = turn(0) + [{"role": "user", "content": "last"}]
        self.svc.drop_oldest = False
        before, _ = self.prompt(messages)
        self.svc.drop_oldest = True
        after, _ = self.prompt(messages)
        self.assertEqual(before, after)

    def test_tool_results_go_with_their_turn(self):
        self.svc.drop_oldest = True
        call = {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c0", "function": {"name": "read", "arguments": {}}}]}
        messages = ([{"role": "user", "content": "question 0 " + "q" * 1000}, call,
                     {"role": "tool", "tool_call_id": "c0", "content": "result 0 " + "r" * 1500},
                     {"role": "assistant", "content": "answer 0"}] +
                    [m for i in range(1, 3) for m in turn(i)] + [{"role": "user", "content": "last"}])
        text, _ = self.prompt(messages)
        self.assertNotIn("result 0", text)
        self.assertNotIn("question 0", text)
        self.assertIn("last", text)

    def test_a_last_turn_too_long_is_still_refused(self):
        self.svc.drop_oldest = True
        messages = turn(0) + [{"role": "user", "content": "x" * (CTX + 100)}]
        with self.assertRaisesRegex(ValueError, "no room to answer"):
            self.svc.prepare(messages, None, {}, None)

    def test_a_large_max_tokens_is_shortened(self):
        self.svc.drop_oldest = True
        messages = [m for i in range(4) for m in turn(i)] + [{"role": "user", "content": "last"}]
        _, max_new = self.prompt(messages, 32000)
        self.assertLess(max_new, CTX)
        self.assertGreaterEqual(max_new, CTX // 8)


if __name__ == "__main__":
    unittest.main()
