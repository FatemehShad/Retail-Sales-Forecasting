"""Fetch checksum-verified Delta jars for local Spark tests (Databricks includes Delta)."""

import argparse
import hashlib
import urllib.request
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    directory = Path(args.output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    for artifact in ["delta-spark_2.12", "delta-storage"]:
        name = f"{artifact}-3.3.2.jar"
        url = f"https://repo.maven.apache.org/maven2/io/delta/{artifact}/3.3.2/{name}"
        # Maven Central publishes SHA-1 sidecars for these artifacts; TLS remains enabled.
        with urllib.request.urlopen(url + ".sha1", timeout=60) as response:
            expected = response.read().decode().strip().split()[0]
        path = directory / name
        if path.exists() and hashlib.sha1(path.read_bytes()).hexdigest() == expected:
            continue
        with urllib.request.urlopen(url, timeout=60) as response:
            payload = response.read()
        if hashlib.sha1(payload).hexdigest() != expected:
            raise RuntimeError(f"Maven checksum mismatch for {name}")
        path.write_bytes(payload)
    print(
        ",".join(
            str(directory / f"{artifact}-3.3.2.jar")
            for artifact in ["delta-spark_2.12", "delta-storage"]
        )
    )


if __name__ == "__main__":
    main()
