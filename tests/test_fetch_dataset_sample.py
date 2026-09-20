"""Offline tests for the ModelScope prefix fetcher.

The network call itself is not exercised here; the parsing of a byte prefix is,
because that is where a truncated download turns into corrupt training data.
"""

import io
import json

from tools.fetch_dataset_sample import complete_records, download_url, fetch_prefix, write_jsonl


def encode(records: list[dict]) -> bytes:
    return "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records).encode("utf-8")


def test_whole_lines_are_parsed():
    payload = encode([{"text": "one"}, {"text": "two"}])
    records, consumed = complete_records(payload, limit=10)
    assert [record["text"] for record in records] == ["one", "two"]
    assert consumed == len(payload) - 1  # trailing newline is not part of the body


def test_truncated_final_line_is_discarded():
    payload = encode([{"text": "keep"}]) + b'{"text": "cut of'
    records, _ = complete_records(payload, limit=10)
    assert [record["text"] for record in records] == ["keep"]


def test_prefix_cut_mid_multibyte_character_does_not_raise():
    # A range request ends at an arbitrary byte, which for CJK text routinely
    # lands inside a character.
    payload = encode([{"text": "完整"}]) + '{"text": "截断'.encode()[:-1]
    records, _ = complete_records(payload, limit=10)
    assert [record["text"] for record in records] == ["完整"]


def test_limit_stops_early():
    payload = encode([{"text": str(index)} for index in range(50)])
    records, _ = complete_records(payload, limit=5)
    assert len(records) == 5


def test_empty_lines_are_ignored():
    payload = b'{"text": "a"}\n\n\n{"text": "b"}\n'
    records, _ = complete_records(payload, limit=10)
    assert [record["text"] for record in records] == ["a", "b"]


def test_payload_without_any_newline_yields_nothing():
    records, consumed = complete_records(b'{"text": "no newline yet', limit=10)
    assert records == []
    assert consumed >= 0


def test_download_url_targets_the_dataset_repo():
    url = download_url("gongjy/minimind_dataset", "pretrain_t2t_mini.jsonl", "master")
    assert url == (
        "https://www.modelscope.cn/api/v1/datasets/gongjy/minimind_dataset/repo"
        "?Revision=master&FilePath=pretrain_t2t_mini.jsonl"
    )


def test_fetch_prefix_stays_bounded_when_server_ignores_range(monkeypatch):
    class Response(io.BytesIO):
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda request, timeout: Response(b"0123456789"),
    )
    assert fetch_prefix("https://example.invalid/data", byte_count=4, timeout=1) == b"0123"


def test_write_jsonl_round_trips_unicode(tmp_path):
    path = tmp_path / "nested" / "out.jsonl"
    write_jsonl(path, [{"text": "中文"}, {"text": "ascii"}])
    lines = path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["text"] for line in lines] == ["中文", "ascii"]
    assert "\\u" not in path.read_text(encoding="utf-8")
