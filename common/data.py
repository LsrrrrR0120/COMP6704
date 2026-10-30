"""GPU-resident datasets: CIFAR-10 (with on-GPU augmentation) and character-level text.

Keeping the whole dataset on the GPU avoids DataLoader worker overhead on
Windows and makes the per-step time dominated by the model and the optimizer.
"""
import os
import urllib.request
import zipfile

import numpy as np
import torch
import torch.nn.functional as F

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)
TINY_SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
ENWIK8_URL = "http://mattmahoney.net/dc/enwik8.zip"


# ----------------------------------------------------------------------------- CIFAR-10
def load_cifar10(root, device, synthetic=False):
    """Return dict(train=(x_uint8 NCHW, y), test=(...)) on `device`."""
    if synthetic:  # random data with CIFAR shapes, for smoke tests without downloading
        g = torch.Generator().manual_seed(0)
        mk = lambda n: (torch.randint(0, 256, (n, 3, 32, 32), generator=g, dtype=torch.uint8),
                        torch.randint(0, 10, (n,), generator=g))
        (xtr, ytr), (xte, yte) = mk(2048), mk(512)
    else:
        cache = os.path.join(root, "cifar10_uint8.pt")
        if not os.path.exists(cache):
            from torchvision.datasets import CIFAR10
            tr = CIFAR10(root, train=True, download=True)
            te = CIFAR10(root, train=False, download=True)
            to_t = lambda d: (torch.from_numpy(d.data).permute(0, 3, 1, 2).contiguous(),
                              torch.tensor(d.targets))
            torch.save(dict(train=to_t(tr), test=to_t(te)), cache)
        d = torch.load(cache)
        (xtr, ytr), (xte, yte) = d["train"], d["test"]
    return dict(train=(xtr.to(device), ytr.to(device)), test=(xte.to(device), yte.to(device)))


def normalize_cifar(x_uint8):
    mean = torch.tensor(CIFAR_MEAN, device=x_uint8.device).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR_STD, device=x_uint8.device).view(1, 3, 1, 1)
    return (x_uint8.float() / 255.0 - mean) / std


def augment_cifar(x, pad=4):
    """Random crop (reflect padding) + horizontal flip, done on the GPU."""
    n, c, h, w = x.shape
    xp = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    i = torch.randint(0, 2 * pad + 1, (n,), device=x.device)
    j = torch.randint(0, 2 * pad + 1, (n,), device=x.device)
    rows = (i[:, None] + torch.arange(h, device=x.device))[:, None, :, None]
    cols = (j[:, None] + torch.arange(w, device=x.device))[:, None, None, :]
    out = xp[torch.arange(n, device=x.device)[:, None, None, None],
             torch.arange(c, device=x.device)[None, :, None, None], rows, cols]
    flip = torch.rand(n, device=x.device) < 0.5
    return torch.where(flip[:, None, None, None], out.flip(-1), out)


def cifar_batches(x, y, batch_size, shuffle=True, augment=False, drop_last=True):
    n = x.size(0)
    idx = torch.randperm(n, device=x.device) if shuffle else torch.arange(n, device=x.device)
    stop = n - n % batch_size if drop_last else n
    for s in range(0, stop, batch_size):
        b = idx[s:s + batch_size]
        xb = normalize_cifar(x[b])
        if augment:
            xb = augment_cifar(xb)
        yield xb, y[b]


# ----------------------------------------------------------------------------- text
CORPORA = {
    # name: (default path, download URL)
    "shakespeare": ("data/tinyshakespeare.txt", TINY_SHAKESPEARE_URL),
    "enwik8": ("data/enwik8", ENWIK8_URL),
}


def _download_corpus(path, url):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if url.endswith(".zip"):
        tmp = path + ".zip"
        urllib.request.urlretrieve(url, tmp)
        with zipfile.ZipFile(tmp) as z:
            z.extract(os.path.basename(path), os.path.dirname(os.path.abspath(path)))
        os.remove(tmp)
    else:
        urllib.request.urlretrieve(url, path)


class CharDataset:
    """Byte-level corpus split 90/10 into train/val, stored as uint8 on `device`.

    Tokens are the distinct bytes of the file (Tiny Shakespeare is ASCII, so bytes
    are characters; enwik8 is the standard 100 MB byte-level benchmark).
    """

    def __init__(self, path, device, synthetic=False, download=False, url=TINY_SHAKESPEARE_URL):
        if synthetic:
            rng = np.random.default_rng(0)
            raw = rng.choice(np.frombuffer(b"abcdefgh ijk\n", dtype=np.uint8), size=200_000)
        else:
            if not os.path.exists(path):
                if not download:
                    raise FileNotFoundError(f"{path} not found; pass --download to fetch it")
                _download_corpus(path, url)
            with open(path, "rb") as f:
                raw = np.frombuffer(f.read(), dtype=np.uint8)
        byte_values = np.unique(raw)
        self.vocab_size = len(byte_values)
        lut = np.zeros(256, dtype=np.uint8)
        lut[byte_values] = np.arange(len(byte_values), dtype=np.uint8)
        data = torch.from_numpy(lut[raw])
        n = int(0.9 * len(data))
        self.train, self.val = data[:n].to(device), data[n:].to(device)

    def get_batch(self, split, batch_size, block_size, generator=None):
        data = self.train if split == "train" else self.val
        ix = torch.randint(len(data) - block_size - 1, (batch_size,), device=data.device, generator=generator)
        offs = torch.arange(block_size, device=data.device)
        x = data[ix[:, None] + offs].long()
        y = data[ix[:, None] + offs + 1].long()
        return x, y
