from pathlib import Path
from typing import NamedTuple

import argparse
import json
import re
import zlib

FILE_PATH = Path(__file__)

PATIENT_ID_RE = re.compile(r"# (HMU\_\d{3}\_[A-Z]{2})")
NEGATIVE_TERMS = (
    "torn",
    "deform",
    "missing",
    "too",
    "very",
    "fucked",
    "weird",
    "uneven",
    "maybe",
    "error",
    "light",
    "dark",
    "folds",
)


class Correspondence(NamedTuple):
    patient_id: str
    histology_id: str
    t2w_slice_index: int

    @staticmethod
    def to_bytes(c: "Correspondence") -> str:
        return json.dumps(tuple(c), separators=(",", ":"))


def extract_correspondences_from_markdown(
    md: Path, verbose: bool
) -> list[Correspondence]:
    corrs: list[Correspondence] = []
    with open(md) as f:
        # Obtain patient ID.
        header = f.readline().strip()
        if (m := PATIENT_ID_RE.match(header)) is None:
            raise ValueError("could not read patient ID!")
        patient_id = m.group(1)
        # Identify correspondences table.
        # > Look for header.
        while (line := f.readline()) and "**Histology**" not in line:
            continue
        if not line:
            # No correspondences' table found.
            return []
        # > Find column indices.
        columns = tuple(
            map(
                lambda x: x.strip().strip("**").lower(),
                line.strip("|").split("|"),
            )
        )
        hist_index = columns.index("histology")
        comment_index = hist_index + 1
        t2w_index = columns.index("in vivo t2w")
        # > Consume separator.
        _ = f.readline()
        # > Read entries.
        while entry := f.readline().strip():
            parts = tuple(
                map(
                    lambda x: x.strip().strip("**"), entry.strip("|").split("|")
                )
            )
            # >> Check histology value.
            hist = parts[hist_index]
            if hist.lower() == "total":
                # End of table.
                break
            # >> Check T2W slice index.
            t2w_slice_index = parts[t2w_index]
            if not t2w_slice_index:
                continue
            try:
                t2w_slice_index = int(t2w_slice_index)
            except ValueError:
                if verbose:
                    print(
                        f"[{patient_id=}] Ambiguous Slice Index: {t2w_slice_index}"
                    )
                continue
            # >> Check comment.
            comment = parts[comment_index].lower()
            if any(x in comment for x in NEGATIVE_TERMS):
                print(f"[{patient_id=}] Negative: {comment!r}")
                continue
            if verbose and comment:
                print(f"[{patient_id=}] Comment: {comment!r}")
            # >> Add correspondence!
            corrs.append(
                Correspondence(
                    patient_id=patient_id,
                    histology_id=hist,
                    t2w_slice_index=int(t2w_slice_index),
                )
            )
    return corrs


def serialise(corrs: list[Correspondence]) -> bytes:
    serial = json.dumps(
        corrs, default=Correspondence.to_bytes, separators=(",", ":")
    )
    return zlib.compress(serial.encode("utf-8"))


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        usage="     python extract.py [markdown]\n"
        "Example:    python extract.py HMU_*.md",
    )
    parser.add_argument(
        "markdown",
        nargs="+",
        type=Path,
        help="Markdown file(s) to process",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="ssregpros/datasets/histo_mri/corrs.bin",
        type=Path,
        help="Output file to write the zlib compressed correspondences to",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print debugging messages",
        default=False,
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    corrs = [
        corr
        for md in args.markdown
        for corr in extract_correspondences_from_markdown(
            md, verbose=args.verbose
        )
    ]
    num_patients = len(set(c.patient_id for c in corrs))
    buf = serialise(corrs)
    if not args.output.parent.exists():
        args.output = FILE_PATH / "corrs.bin"
    with open(args.output, "wb") as out:
        out.write(buf)
    print(
        f"{len(corrs)} correspondences ({num_patients} patients) "
        f"written to file {str(args.output)} ({len(buf)} bytes)."
    )


if __name__ == "__main__":
    main()
