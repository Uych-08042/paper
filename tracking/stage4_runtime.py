"""Frozen SGLA operations for Stage-4 profiling and feature collection.

PyTorch and the legacy tracker are imported only when constructing the runtime,
so the CPU-only manifest, cache checker and command help need no tracker deps.
"""

from pathlib import Path

from stage4_common import file_hash


class Runtime:
    def __init__(self, parameter="deit_distilled", gpu=0, checkpoint=None):
        import stage1_validate as stage1

        self.s1 = stage1
        self.torch = stage1.torch
        if not self.torch.cuda.is_available():
            raise RuntimeError("The original tracker requires CUDA")
        self.torch.cuda.set_device(gpu)
        self.info = stage1.Tracker("sglatrack", parameter, "stage4")
        params = self.info.get_parameters()
        params.debug = 0
        if checkpoint:
            params.checkpoint = str(Path(checkpoint).expanduser().resolve())
        if not Path(params.checkpoint).is_file():
            raise FileNotFoundError("Checkpoint missing: {} (or pass --checkpoint)".format(params.checkpoint))
        self.tracker = self.info.create_tracker(params)
        self.net = self.tracker.network.eval()
        self.net.requires_grad_(False)
        self.backbone = self.net.backbone
        if stage1.start_layer != 5 or stage1.enabled_layer_num != 1 or len(self.backbone.blocks) != 12:
            raise ValueError("Stage-4 requires L1-L6 plus one independent L7-L12 candidate")
        if self.backbone.cat_mode != "direct":
            raise ValueError("Stage-4 feature cache currently supports CAT_MODE=direct only")
        self.lens_z = self.backbone.pos_embed_z.shape[1]
        self.lens_x = self.backbone.pos_embed_x.shape[1]
        self.search_side = int(self.lens_x ** 0.5)
        if self.search_side ** 2 != self.lens_x:
            raise ValueError("Expected a square search token grid")
        self.identity = {
            "parameter": parameter, "checkpoint_sha256": file_hash(params.checkpoint),
            "config": params.cfg, "torch": str(self.torch.__version__),
            "cuda": str(self.torch.version.cuda), "device": self.torch.cuda.get_device_name(gpu),
            "lens_z": int(self.lens_z), "lens_x": int(self.lens_x),
            "embed_dim": int(self.backbone.pos_embed_x.shape[2]),
        }

    def read(self, path):
        return self.info._read_image(str(path))

    def initialize(self, image, box):
        self.tracker.initialize(image, {"init_bbox": list(box)})

    def prepare(self, image):
        patch, factor, mask = self.s1.sample_target(
            image, self.tracker.state, self.tracker.params.search_factor,
            output_sz=self.tracker.params.search_size,
        )
        return self.tracker.preprocessor.process(patch, mask).tensors, factor

    def prefix(self, search):
        bb = self.backbone
        z = bb.patch_embed(self.tracker.z_dict1.tensors) + bb.pos_embed_z
        x = bb.patch_embed(search) + bb.pos_embed_x
        tokens = bb.pos_drop(self.s1.combine_tokens(z, x, mode=bb.cat_mode))
        for block in bb.blocks[:6]:
            tokens = block(tokens)
        return tokens

    def head(self, tokens):
        tokens = self.s1.recover_tokens(tokens, self.lens_z, self.lens_x, mode=self.backbone.cat_mode)
        return self.net.forward_head(self.backbone.norm(tokens), None)

    def selector(self, tokens):
        return self.backbone.MLP(tokens[:, :, 0].clone())

    def selected_layer(self, probabilities):
        return int(self.torch.topk(probabilities, 1, dim=1).indices[0, 0].item()) + 7

    def candidate(self, tokens, layer):
        if layer not in range(7, 13):
            raise ValueError("Candidate must be L7-L12")
        return self.head(self.backbone.blocks[layer - 1](tokens.clone()))

    def pool(self, tokens, grid=4):
        if not 1 <= grid <= self.search_side:
            raise ValueError("Pool grid outside search token dimensions")
        # Keep all embedding channels, unlike the original selector's [:,:,0].
        template = tokens[:, :self.lens_z]
        search = tokens[:, self.lens_z:].transpose(1, 2).reshape(
            tokens.shape[0], tokens.shape[2], self.search_side, self.search_side)
        search = self.torch.nn.functional.adaptive_avg_pool2d(search, (grid, grid)).flatten(1)
        return self.torch.cat((template.mean(1), template.max(1).values, search), dim=1)

    def decode(self, output, factor, image):
        return self.s1._decode_box(self.tracker, output, factor, image.shape[0], image.shape[1])

    def run_image(self, image, mode):
        search, factor = self.prepare(image)
        tokens = self.prefix(search)
        if mode == "fast":
            return self.decode(self.head(tokens), factor, image)
        if mode == "fast_then_sgla":
            # Measure the real overhead of a score-map-based gate's Fast head.
            self.decode(self.head(tokens), factor, image)
        elif mode == "fixed_l9":
            return self.decode(self.candidate(tokens, 9), factor, image)
        elif mode != "sgla":
            raise ValueError("Unknown measured mode: {}".format(mode))
        layer = self.selected_layer(self.selector(tokens))
        return self.decode(self.candidate(tokens, layer), factor, image)

    def verify(self, search, scripted):
        self.s1._verify_sgla_output(self.tracker, self.tracker.z_dict1.tensors, search, scripted)


def add_runtime_arguments(parser):
    parser.add_argument("--parameter", default="deit_distilled")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--gpu", type=int, default=0)
