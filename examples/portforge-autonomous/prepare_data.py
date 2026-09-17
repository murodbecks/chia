"""Rebuild the verified public FLORES corpus without importing an earlier port."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import urllib.request
from common import digest, write_json
from suite import CORPUS_SHA256, LANGUAGES

URL = "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz"
ARCHIVE_SHA256 = "b8b0b76783024b85797e5cc75064eb83fc5288b41e9654dabc7be6ae944011f6"


def prepare(archive, output):
    archive, output = Path(archive), Path(output)
    sha = hashlib.sha256()
    with archive.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            sha.update(block)
    if sha.hexdigest() != ARCHIVE_SHA256:
        raise ValueError("official FLORES archive checksum mismatch")
    corpus = {}
    with tarfile.open(archive, "r:gz") as tar:
        for lang in ("eng_Latn", *LANGUAGES):
            name = "./flores200_dataset/devtest/" + lang + ".devtest"
            member = tar.getmember(name)
            if not member.isfile() or member.size > 5_000_000:
                raise ValueError("unexpected dataset member")
            corpus[lang] = tar.extractfile(member).read().decode("utf-8").splitlines()
    if digest(corpus) != CORPUS_SHA256:
        raise ValueError("decoded corpus identity mismatch")
    write_json(output, corpus)
    write_json(output.with_name("provenance.json"), {"url": URL, "archive_sha256": ARCHIVE_SHA256,
        "corpus_sha256": CORPUS_SHA256, "license": "CC-BY-SA-4.0", "split": "devtest",
        "languages": list(corpus), "aligned_rows": 1012})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    if args.download and not args.archive.exists():
        args.archive.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(URL, timeout=120) as response, args.archive.open("wb") as f:
            while block := response.read(1024 * 1024):
                f.write(block)
    prepare(args.archive, args.output)
