import json

import torch
from karume.shards import piece_key
from safetensors.torch import save_file
from weights import StoredWeights


def test_split_int4_restores_signed_values_and_group_scales(tmp_path):
    graph = {
        "values": {"w": {"shape": [2, 16]}},
        "initializers": {
            "w": {
                "tensor": "model.weight",
                "storage": {"dtype": "i4", "scale": "scale", "group_size": 16},
            }
        },
    }
    first = tmp_path / "graph.safetensors"
    save_file(
        {"scale": torch.tensor([[0.5], [2.0]])}, first, metadata={"karume_ir": json.dumps(graph)}
    )
    # 下位 nibble 1→−7、上位 nibble 15→7。2行目は下位8→0、上位9→1。
    pieces = []
    for index, byte in [(1, 0xF1), (2, 0x98)]:
        file = tmp_path / f"piece-{index}.safetensors"
        save_file(
            {piece_key("model.weight", index, 2): torch.full((8,), byte, dtype=torch.uint8)}, file
        )
        pieces.append(file)
    reader = StoredWeights([first, *reversed(pieces)])
    value = reader.dequantized("w", graph["initializers"]["w"])
    expected = torch.tensor([[-3.5, 3.5] * 8, [0.0, 2.0] * 8])
    assert torch.equal(value, expected)
