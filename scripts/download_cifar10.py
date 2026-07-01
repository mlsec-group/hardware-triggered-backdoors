#!/usr/bin/env python3

import hashlib
import tarfile
import tempfile
import urllib.request
from pathlib import Path

CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR10_MD5 = "c58f30108f718f92721af3b95e74349a"


def digest(path: Path, algorithm: str) -> str:
    h = hashlib.new(algorithm)
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    out_dir = Path("data/cifar10")
    out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="cifar10_") as tmp:
        archive = Path(tmp) / "cifar-10-python.tar.gz"
        print(f"Downloading {CIFAR10_URL}")
        urllib.request.urlretrieve(CIFAR10_URL, archive)

        md5 = digest(archive, "md5")
        if md5 != CIFAR10_MD5:
            raise RuntimeError(f"CIFAR-10 MD5 mismatch: got {md5}, expected {CIFAR10_MD5}")

        with tarfile.open(archive, "r:gz") as tar:
            members = [
                m
                for m in tar.getmembers()
                if m.isfile()
                and Path(m.name).name
                in {
                    "data_batch_1",
                    "data_batch_2",
                    "data_batch_3",
                    "data_batch_4",
                    "data_batch_5",
                    "test_batch",
                    "batches.meta",
                }
            ]
            for member in members:
                member.name = Path(member.name).name
                tar.extract(member, out_dir)

    print(f"CIFAR-10 batches written to {out_dir}")


if __name__ == "__main__":
    main()
