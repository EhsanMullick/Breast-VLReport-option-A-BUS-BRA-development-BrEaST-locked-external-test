"""Breast-VLReport and baselines. One ReportModel class; flags switch lesion / attribute / contrastive parts,
so baselines and ablations (G: direct image->report, H: +lesion, I: +attributes, A: full) share one code path."""
import math, random
import torch, torch.nn as nn, torch.nn.functional as F
import timm

ENCODERS = {"swin": "swin_tiny_patch4_window7_224", "vit": "vit_base_patch16_224", "cnn": "resnet50"}


class Encoder(nn.Module):
    def __init__(self, kind, pretrained=True):
        super().__init__()
        self.m = timm.create_model(ENCODERS[kind], pretrained=pretrained, num_classes=0, global_pool="")
        self.dim = self.m.num_features

    def forward(self, x):
        f = self.m.forward_features(x)
        if f.dim() == 4:  # NHWC (swin) or NCHW (resnet)
            if f.shape[-1] == self.dim and f.shape[1] != self.dim:
                f = f.permute(0, 3, 1, 2)
            B, C, H, W = f.shape
            return f.flatten(2).transpose(1, 2), (H, W)
        N = f.shape[1]; g = math.isqrt(N)  # ViT: drop cls/prefix tokens
        return f[:, N - g * g:], (g, g)


class SegHead(nn.Module):
    def __init__(self, c):
        super().__init__()
        blk = lambda i, o: nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(True))
        self.net = nn.Sequential(blk(c, 128), nn.Upsample(scale_factor=2), blk(128, 64), nn.Upsample(scale_factor=2),
                                 blk(64, 32), nn.Upsample(scale_factor=2), nn.Conv2d(32, 1, 1))

    def forward(self, fmap):
        return F.interpolate(self.net(fmap), size=(224, 224), mode="bilinear", align_corners=False)


# ---- decoders: forward(ids, memory) -> logits; embed(ids) -> token embeddings ------------------------------
class TfmDecoder(nn.Module):
    def __init__(self, V, d, layers=4, heads=8, max_len=128):
        super().__init__()
        self.emb, self.pos = nn.Embedding(V, d), nn.Parameter(torch.zeros(max_len, d))
        self.dec = nn.TransformerDecoder(nn.TransformerDecoderLayer(d, heads, 4 * d, 0.1, batch_first=True, norm_first=True), layers)
        self.norm, self.out = nn.LayerNorm(d), nn.Linear(d, V)

    def embed(self, ids): return self.emb(ids)

    def forward(self, ids, mem):
        L = ids.size(1)
        mask = torch.triu(torch.full((L, L), float("-inf"), device=ids.device), 1)
        return self.out(self.norm(self.dec(self.emb(ids) + self.pos[:L], mem, tgt_mask=mask)))


class LSTMDecoder(nn.Module):  # classical CNN-RNN baseline with dot-product attention over visual tokens
    def __init__(self, V, d):
        super().__init__()
        self.emb, self.cell = nn.Embedding(V, d), nn.LSTMCell(2 * d, d)
        self.init_h, self.out = nn.Linear(d, 2 * d), nn.Linear(2 * d, V)

    def embed(self, ids): return self.emb(ids)

    def forward(self, ids, mem):
        h, c = self.init_h(mem.mean(1)).chunk(2, -1)
        outs = []
        for t in range(ids.size(1)):
            a = torch.softmax((mem @ h.unsqueeze(-1)).squeeze(-1) / math.sqrt(h.size(-1)), -1)
            ctx = (a.unsqueeze(-1) * mem).sum(1)
            h, c = self.cell(torch.cat([self.emb(ids[:, t]), ctx], -1), (h, c))
            outs.append(self.out(torch.cat([h, ctx], -1)))
        return torch.stack(outs, 1)


class GPT2Decoder(nn.Module):  # pretrained GPT-2 + (newly initialised) cross-attention to visual/clinical tokens
    def __init__(self, name="gpt2"):
        super().__init__()
        from transformers import GPT2Config, GPT2LMHeadModel
        cfg = GPT2Config.from_pretrained(name); cfg.add_cross_attention = True
        self.m, self.d = GPT2LMHeadModel.from_pretrained(name, config=cfg), cfg.n_embd

    def embed(self, ids): return self.m.transformer.wte(ids)
    def forward(self, ids, mem): return self.m(input_ids=ids, encoder_hidden_states=mem).logits


class ReportModel(nn.Module):
    def __init__(self, tok, schema, enc="swin", dec="tfm", lesion=True, attr=True, contrastive=True,
                 d=512, pretrained=True, gpt2="gpt2"):
        super().__init__()
        self.tok, self.schema, self.names = tok, schema, list(schema)
        self.use_lesion, self.use_attr, self.use_con = lesion, attr, contrastive
        self.enc = Encoder(enc, pretrained)
        if dec == "gpt2":
            self.dec = GPT2Decoder(gpt2); d = self.dec.d
        else:
            self.dec = LSTMDecoder(tok.V, d) if dec == "lstm" else TfmDecoder(tok.V, d)
        self.d = d
        self.proj = nn.Linear(self.enc.dim, d)
        if lesion:
            self.seg = SegHead(self.enc.dim)
            self.lesion_embed = nn.Parameter(torch.zeros(d))
            self.extra = nn.Parameter(torch.zeros(2, d))  # type embeddings for [global, lesion] tokens
        if attr:
            assert lesion, "attribute module reads the lesion vector"
            self.heads = nn.ModuleDict({a: nn.Linear(2 * d, len(v)) for a, v in schema.items()})
            self.attr_emb = nn.ModuleDict({a: nn.Linear(len(v), d, bias=False) for a, v in schema.items()})
        if contrastive:
            self.pi, self.pt, self.logit_scale = nn.Linear(d, 256), nn.Linear(d, 256), nn.Parameter(torch.tensor(2.6))

    def encode(self, img, gt_mask=None, p_gt=0.5):
        tok, (h, w) = self.enc(img)
        B, N, _ = tok.shape
        mem, out = self.proj(tok), {}
        if not self.use_lesion:
            return mem, out, mem.mean(1)
        fmap = tok.transpose(1, 2).reshape(B, -1, h, w)
        out["seg"] = self.seg(fmap)
        m = torch.sigmoid(out["seg"])
        if self.training and gt_mask is not None and random.random() < p_gt:
            m = gt_mask
        mt = F.adaptive_avg_pool2d(m, (h, w)).flatten(1).unsqueeze(-1)  # B,N,1
        lesion = (mt * mem).sum(1) / (mt.sum(1) + 1e-6)
        glob = mem.mean(1)
        mem = mem + mt * self.lesion_embed
        extra = [glob + self.extra[0], lesion + self.extra[1]]
        if self.use_attr:
            z = torch.cat([glob, lesion], -1)
            out["attr"] = {a: self.heads[a](z) for a in self.names}
            extra += [self.attr_emb[a](F.softmax(out["attr"][a], -1)) for a in self.names]
        return torch.cat([torch.stack(extra, 1), mem], 1), out, lesion

    def forward(self, batch):
        mem, out, vis = self.encode(batch["img"], batch.get("mask"))
        out["logits"] = self.dec(batch["ids"][:, :-1], mem)
        if self.use_con:
            ti = F.normalize(self.pt(self.dec.embed(batch["ids"]).mean(1)), dim=-1)
            vi = F.normalize(self.pi(vis), dim=-1)
            out["sim"] = self.logit_scale.exp().clamp(max=100) * vi @ ti.t()
        return out

    @torch.no_grad()
    def generate(self, img, max_len=None):
        self.eval()
        mem, out, _ = self.encode(img)
        B, L = img.size(0), max_len or self.tok.max_len
        ids = torch.full((B, 1), self.tok.bos, device=img.device)
        done = torch.zeros(B, dtype=torch.bool, device=img.device)
        for _ in range(L - 1):
            nxt = self.dec(ids, mem)[:, -1].argmax(-1)
            nxt = torch.where(done, torch.full_like(nxt, self.tok.pad), nxt)
            ids = torch.cat([ids, nxt[:, None]], 1)
            done |= nxt == self.tok.eos
            if done.all(): break
        return ids, out


def compute_losses(model, batch, out):
    """Returns dict of the individual losses (weights applied by the trainer)."""
    ids = batch["ids"]
    tgt, eos = ids[:, 1:], model.tok.eos
    seen_eos = (tgt == eos).cumsum(1) - (tgt == eos).long()   # positions up to and including first eos are valid
    valid = seen_eos == 0
    L = {"ce": F.cross_entropy(out["logits"][valid], tgt[valid])}
    if "seg" in out:
        p, g = torch.sigmoid(out["seg"]), batch["mask"]
        dice = 1 - (2 * (p * g).sum((1, 2, 3)) + 1) / (p.sum((1, 2, 3)) + g.sum((1, 2, 3)) + 1)
        L["seg"] = dice.mean() + F.binary_cross_entropy_with_logits(out["seg"], g)
    if "attr" in out:
        ls = [F.cross_entropy(out["attr"][a], batch["labels"][:, i], ignore_index=-100)
              for i, a in enumerate(model.names) if (batch["labels"][:, i] != -100).any()]
        L["attr"] = torch.stack(ls).mean()
    if "sim" in out:  # multi-positive InfoNCE: identical reports count as positives (templated text has many)
        pos = (ids[:, None, :] == ids[None, :, :]).all(-1).float()
        for s in (out["sim"], out["sim"].t()):
            lp = s.log_softmax(1)
            L["con"] = L.get("con", 0) + 0.5 * -((lp * pos).sum(1) / pos.sum(1)).mean()
    return L
