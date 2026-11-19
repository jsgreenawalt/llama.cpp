from concurrent.futures import ThreadPoolExecutor
import time

import pytest
from utils import *

server = ServerPreset.tinyllama2()


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.server_slots = True
    server.n_slots = 2
    server.kv_unified = True
    server.cache_ram = 0
    server.no_cache_idle_slots = True
    server.temperature = 0.0


def test_slot_copy():
    global server
    server.start()

    prefix = "The quick brown fox jumps over the lazy dog. " * 10

    res = server.make_request("POST", "/tokenize", data={
        "content": prefix,
        "add_special": True,
    })
    assert res.status_code == 200
    prefix_tokens = res.body["tokens"]

    res = server.make_request("POST", "/tokenize", data={
        "content": "The destination continuation is independent.",
    })
    assert res.status_code == 200
    destination_tokens = res.body["tokens"]

    res = server.make_request("POST", "/tokenize", data={
        "content": "The source continuation is still independent.",
    })
    assert res.status_code == 200
    source_tokens = res.body["tokens"]

    res = server.make_request("POST", "/completion", data={
        "prompt": prefix_tokens,
        "id_slot": 0,
        "n_predict": 0,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    n_prefill = len(prefix_tokens)
    assert res.body["timings"]["prompt_n"] == n_prefill

    res = server.make_request("POST", "/slots/0?action=copy&id_dst=1")
    assert res.status_code == 200
    assert res.body["id_src"] == 0
    assert res.body["id_dst"] == 1
    assert res.body["n_tokens"] == n_prefill

    res = server.make_request("POST", "/completion", data={
        "prompt": prefix_tokens + destination_tokens,
        "id_slot": 1,
        "n_predict": 0,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert res.body["timings"]["cache_n"] == n_prefill
    assert res.body["timings"]["prompt_n"] < n_prefill

    res = server.make_request("POST", "/completion", data={
        "prompt": prefix_tokens + source_tokens,
        "id_slot": 0,
        "n_predict": 0,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert res.body["timings"]["cache_n"] == n_prefill
    assert res.body["timings"]["prompt_n"] < n_prefill


@pytest.mark.parametrize("path", [
    "/slots/2?action=copy&id_dst=1",
    "/slots/0?action=copy&id_dst=2",
    "/slots/0?action=copy&id_dst=0",
    "/slots/0?action=copy&id_dst=invalid",
])
def test_slot_copy_rejects_invalid_ids(path):
    global server
    server.start()

    res = server.make_request("POST", path)
    assert res.status_code == 400


def test_slot_copy_rejects_processing_slot():
    global server
    server.start()

    prompt = "The quick brown fox jumps over the lazy dog. " * 15

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(server.make_request, "POST", "/completion", {
            "prompt": prompt,
            "id_slot": 0,
            "n_predict": 64,
            "cache_prompt": True,
        })

        for _ in range(100):
            res = server.make_request("GET", "/slots")
            if res.status_code == 200:
                slot = next(slot for slot in res.body if slot["id"] == 0)
                if slot["is_processing"]:
                    break
            time.sleep(0.01)
        else:
            pytest.fail("slot 0 did not start processing")

        res = server.make_request("POST", "/slots/0?action=copy&id_dst=1")
        assert res.status_code == 503

        res = future.result()
        assert res.status_code == 200
