"""LFM2.5 behind the server: its pre-tokenizer, its pythonic tool calls, its {% generation %} template and its call
format.

    python -m unittest serve.test_lfm2
"""
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))

import strata_tokenizer as ST  # noqa: E402
from serve.frontend import ChatTemplate, OutputParser, parse_lfm2_tool_calls  # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service  # noqa: E402


class PreTokenizer(unittest.TestCase):
    def test_pieces_and_whole_tokens(self):
        # llama.cpp's LLAMA3 pattern: punctuation keeps the line breaks after it, a tab goes with the word
        tok = ST.Tokenizer(["a", "b", "ab", "!", " "], [], pre="lfm2")
        self.assertEqual(tok._pieces("Hello  world!\n\n\tTabs 12345"),
                         ["Hello", " ", " world", "!\n\n", "\tTabs", " ", "123", "45"])
        # ignore_merges: a piece that is a token is that token, with no merge to make it
        self.assertEqual(tok.encode("ab"), [2])


class ToolCalls(unittest.TestCase):
    SCHEMA = [{"name": "write", "parameters": {"properties": {"path": {"type": "string"}, "n": {"type": "integer"},
                                                              "content": {"type": "string"}}}},
              {"name": "now", "parameters": {"properties": {}}}]
    CONTENT = "it's (a), [b] and \"c\"\nend"
    TEXT = ("thinking</think>I will write it.<|tool_call_start|>[write(path='doc.md', n=3, content='it\\'s (a), "
            "[b] and \"c\"\\nend'), now()]<|tool_call_end|>")

    def events(self, stream_tools, step):
        p = OutputParser(thinking=True, tools=self.SCHEMA, stream_tools=stream_tools, call_format="lfm2")
        evs = []
        for i in range(0, len(self.TEXT), step):
            evs += p.feed(self.TEXT[i:i + step])
        return evs + p.finish()

    def test_calls_are_parsed_however_they_arrive(self):
        for stream_tools in (False, True):
            for step in (1, 7, 10_000):
                with self.subTest(stream_tools=stream_tools, step=step):
                    evs = self.events(stream_tools, step)
                    calls = [(e.call.name, e.call.arguments) for e in evs if e.kind == "tool_call"]
                    self.assertEqual(calls, [("write", {"path": "doc.md", "n": 3, "content": self.CONTENT}),
                                             ("now", {})])
                    self.assertEqual("".join(e.text for e in evs if e.kind == "reasoning"), "thinking")
                    self.assertEqual("".join(e.text for e in evs if e.kind == "content"), "I will write it.")

    def test_streamed_first_call_keeps_its_id(self):
        evs = [e for e in self.events(True, 1) if e.kind.startswith("tool_")]
        self.assertEqual([e.kind for e in evs], ["tool_start", "tool_args", "tool_call", "tool_call"])
        self.assertEqual(evs[0].call.id, evs[2].call.id)

    def test_values(self):
        calls = parse_lfm2_tool_calls("[f(s=\"x\", j={\"a\": true, \"b\": [1, null]}, p={'a': True}, f=-1.5, "
                                      "t=true, n=None, w=bare)]")
        self.assertEqual(calls[0].arguments, {"s": "x", "j": {"a": True, "b": [1, None]}, "p": {"a": True},
                                              "f": -1.5, "t": True, "n": None, "w": "bare"})

    def test_answer_without_thinking_drops_its_first_newline(self):
        # asked not to think, the prompt ends on <think></think> and LFM2 starts its answer on a new line
        p = OutputParser(thinking=False, call_format="lfm2")
        evs = p.feed("\n") + p.feed("Bonjour !\nSalut.") + p.finish()
        self.assertEqual("".join(e.text for e in evs if e.kind == "content"), "Bonjour !\nSalut.")

    def test_a_block_without_a_call_is_text(self):
        p = OutputParser(thinking=False, tools=self.SCHEMA, call_format="lfm2")
        evs = p.feed("x<|tool_call_start|>[]<|tool_call_end|>") + p.finish()
        self.assertEqual("".join(e.text for e in evs if e.kind == "content"),
                         "x<|tool_call_start|>[]<|tool_call_end|>")
        self.assertFalse([e for e in evs if e.kind == "tool_call"])


# the parts of LFM2.5's chat template the server has to meet: {% generation %} around the assistant's turn, tools
# listed in the system turn, calls written as a Python list, and a reply that always opens <think>
TEMPLATE = """{{- bos_token -}}
{%- if tools -%}<|im_start|>system
List of tools: {{ tools | tojson }}<|im_end|>
{% endif -%}
{%- for m in messages -%}<|im_start|>{{ m.role }}
{% if m.role == "assistant" -%}{%- generation -%}{{ m.content }}
{%- for c in m.tool_calls or [] -%}<|tool_call_start|>[{{ c.function.name }}()]<|tool_call_end|>{%- endfor -%}
<|im_end|>
{% endgeneration -%}{%- else -%}{{ m.content }}<|im_end|>
{% endif -%}{%- endfor -%}
{%- if add_generation_prompt -%}<|im_start|>assistant
<think>{%- endif -%}"""


class LfmTokens(ByteTokenizer):
    SPECIALS = ["<|startoftext|>", "<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|tool_call_start|>",
                "<|tool_call_end|>"]
    pre = "lfm2"
    special_ids = {"tokenizer.ggml.bos_token_id": 256, "tokenizer.ggml.eos_token_id": 259}


class ServiceForLfm2(unittest.TestCase):
    def setUp(self):
        tok = LfmTokens()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "chat_template.jinja"
            path.write_text(TEMPLATE, encoding="utf-8")
            self.svc = Service(MockEngine(tok, "hi"), tok, ChatTemplate(path))

    def test_stop_ids_turns_and_call_format(self):
        self.assertEqual(self.svc.call_format, "lfm2")
        self.assertEqual(self.svc.stop_ids, {259, 257})
        self.assertEqual(self.svc.turn_ids, frozenset({258}))

    def test_prompt_with_a_call_in_the_history(self):
        tools = [{"name": "now", "description": "the time", "parameters": {"type": "object", "properties": {}}}]
        messages = [{"role": "user", "content": "time?"},
                    {"role": "assistant", "content": "", "tool_calls": [
                        {"type": "function", "function": {"name": "now", "arguments": "{}"}}]},
                    {"role": "tool", "content": "12:00"}]
        prompt = self.svc.render_prompt(messages, tools, {"enable_thinking": False})
        self.assertTrue(prompt.startswith("<|startoftext|><|im_start|>system\nList of tools: [{\"name\": \"now\""))
        self.assertIn("<|im_start|>assistant\n<|tool_call_start|>[now()]<|tool_call_end|><|im_end|>\n", prompt)
        self.assertTrue(prompt.endswith("<|im_start|>assistant\n<think></think>"))
        self.assertTrue(self.svc.render_prompt(messages[:1], None, {}).endswith("<|im_start|>assistant\n<think>"))


if __name__ == "__main__":
    unittest.main()
