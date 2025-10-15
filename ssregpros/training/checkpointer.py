from ..models.registration import RegistrationNetwork
from .utils import to_jsonable

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import functools
import heapq
import io
import json
import safetensors.torch as st
import shutil
import tarfile
import time
import torch
import zipfile


class _HeapCheckpoint:
    __slots__ = ("epoch", "metric", "filepath", "is_better")

    def __init__(
        self,
        epoch: int,
        metric: Any,
        filepath: Path,
        is_better: Callable[[Any, Any], bool],
    ):
        self.epoch = epoch
        self.metric = metric
        self.filepath = filepath
        self.is_better = is_better

    def __lt__(self, other: "_HeapCheckpoint") -> bool:
        """'Worse' should be considered 'less than' so it gets popped first."""
        if self.is_better(self.metric, other.metric):
            return False
        if self.is_better(other.metric, self.metric):
            return True
        # Tie-breaker: older is worse
        return self.epoch < other.epoch


@dataclass
class CheckpointerConfig:
    root: Path
    top_k: int
    is_better: Callable[[Any, Any], bool]
    metric_name: str
    checkpoints_subdir: str = "checkpoints"


class Checkpointer:
    """
    Single-file checkpoints using ZIP archives:
        - model.safetensors =>  weights on CPU
        - optimiser.pt      =>  torch pickled state, optional
        - meta.json         =>  (epoch, metric, model_config, training_config)

    Maintains a top-k (best) heap using a user-supplied comparator.
    """

    def __init__(self, config: CheckpointerConfig):
        self.config = config
        # Prepare directory and min-heap.
        (self.config.root / config.checkpoints_subdir).mkdir(
            exist_ok=True, parents=True
        )
        self._heap: list[_HeapCheckpoint] = []

    @staticmethod
    def _read_uncompressed(
        checkpoint_path: Path, optimiser: bool, map_location: torch.device
    ) -> tuple[dict[str, torch.Tensor], Any | None, dict]:
        with zipfile.ZipFile(checkpoint_path, mode="r") as zf:
            # Model
            with zf.open("model.safetensors") as f:
                data = f.read()
            state: dict[str, torch.Tensor] = {
                k: v.to(map_location) for k, v in st.load(data).items()
            }
            # Optimiser, optional
            if optimiser:
                with zf.open("optimiser.pt") as f:
                    opt_bytes = f.read()
                opt_state = torch.load(
                    io.BytesIO(opt_bytes), map_location=map_location
                )
            else:
                opt_state = None
            # Metadata
            with zf.open("meta.json") as f:
                meta = json.loads(f.read().decode("utf-8"))
        return state, opt_state, meta

    @staticmethod
    def _read_compressed(
        checkpoint_path: Path, optimiser: bool, map_location: torch.device
    ) -> tuple[dict[str, torch.Tensor], Any | None, dict]:
        with tarfile.open(checkpoint_path, "r:*") as tar:
            # Model
            with tar.extractfile(
                tar.getmember("model.safetensors")
            ) as f:  # pyright: ignore[reportOptionalContextManager]
                data = f.read()
            state: dict[str, torch.Tensor] = {
                k: v.to(map_location) for k, v in st.load(data).items()
            }
            # Optimiser, optional
            if optimiser:
                with tar.extractfile(
                    tar.getmember("optimiser.pt")
                ) as f:  # pyright: ignore[reportOptionalContextManager]
                    opt_bytes = f.read()
                opt_state = torch.load(
                    io.BytesIO(opt_bytes), map_location=map_location
                )
            else:
                opt_state = None
            # Metadata
            with tar.extractfile(
                tar.getmember("meta.json")
            ) as f:  # pyright: ignore[reportOptionalContextManager]
                meta = json.loads(f.read().decode("utf-8"))
        return state, opt_state, meta

    @staticmethod
    def load(
        checkpoint_path: Path,
        model: RegistrationNetwork,
        optimiser: torch.optim.Optimizer | None = None,
        map_location: torch.device = torch.device("cpu"),
    ) -> dict[str, Any]:
        """Loads weights using `safetensors` and optionally optimiser state.
        Returns checkpoint metadata."""
        if ".tar" in checkpoint_path.suffixes:
            state, opt_state, meta = Checkpointer._read_compressed(
                checkpoint_path, optimiser is not None, map_location
            )
        else:
            state, opt_state, meta = Checkpointer._read_uncompressed(
                checkpoint_path, optimiser is not None, map_location
            )
        model.load_state_dict(state, strict=True)
        if optimiser is not None:
            optimiser.load_state_dict(
                opt_state  # pyright: ignore[reportArgumentType]
            )
        return meta

    def _filename_prefix(self) -> str:
        return "model"

    def _ckpt_filename(self, epoch: int, metric: Any) -> str:
        prefix_str = self._filename_prefix()
        metric_str = f"{self.config.metric_name}{metric:.4f}".replace("/", "")
        epoch_str = f"epoch{epoch:04d}"
        suffix = ".ckpt"
        return f'{"_".join((prefix_str, metric_str, epoch_str))}{suffix}'

    def _should_keep(self, epoch: int, metric: Any) -> bool:
        """Would the candidate enter the top-k?"""
        k = self.config.top_k
        if k <= 0:
            return False
        if len(self._heap) < k:
            return True
        worst = self._heap[0]
        if self.config.is_better(metric, worst.metric):
            return True
        if self.config.is_better(worst.metric, metric):
            return False
        # Tie: newer beats older (like _HeapCheckpoint.__lt__)
        return epoch > worst.epoch

    def _save_ckpt(
        self,
        path: Path,
        epoch: int,
        metric: Any,
        model: RegistrationNetwork,
        training_config: dict,
        optimiser: torch.optim.Optimizer | None,
    ):
        """Saves a checkpoint to disk."""
        # Collect model weights on CPU.
        with torch.no_grad():
            tensors_cpu = {
                k: v.detach().cpu() for k, v in model.state_dict().items()
            }
        model_bytes = st.save(tensors_cpu)
        # Save optimiser state to buffer.
        opt_buf = io.BytesIO()
        if optimiser is not None:
            torch.save(optimiser.state_dict(), opt_buf)
        # Save metadata.
        meta = {
            "epoch": epoch,
            self.config.metric_name: metric,
            "model_config": to_jsonable(asdict(model.config)),
            "training_config": to_jsonable(training_config),
        }
        meta_bytes = json.dumps(meta, indent=2).encode("utf-8")
        # Write to ZIP archive, uncompressed.
        with zipfile.ZipFile(
            path, mode="w", compression=zipfile.ZIP_STORED
        ) as zf:
            zf.writestr("model.safetensors", model_bytes)
            zf.writestr("optimiser.pt", opt_buf.getvalue())
            zf.writestr("meta.json", meta_bytes)

    def consider(
        self,
        epoch: int,
        metric: Any,
        model: RegistrationNetwork,
        training_config: dict,
        optimiser: torch.optim.Optimizer | None,
    ) -> Path | None:
        """
        Save a candidate checkpoint and keep only the top-k best on disk.
        Returns the saved path.
        """
        # Skip heavy lifting if not entering top-k.
        if not self._should_keep(epoch, metric):
            return None
        # Save checkpoint.
        filename = self._ckpt_filename(epoch, metric)
        checkpoint_path = (
            self.config.root / self.config.checkpoints_subdir / filename
        )
        self._save_ckpt(
            path=checkpoint_path,
            epoch=epoch,
            metric=metric,
            model=model,
            training_config=training_config,
            optimiser=optimiser,
        )
        # Create heap entry.
        item = _HeapCheckpoint(
            epoch=epoch,
            metric=metric,
            filepath=checkpoint_path,
            is_better=self.config.is_better,
        )
        heapq.heappush(self._heap, item)
        # If we now have more than k, pop worst.
        if len(self._heap) > self.config.top_k:
            worst = heapq.heappop(self._heap)
            worst.filepath.unlink()
        # Return checkpoint path.
        return checkpoint_path

    def save_last(
        self,
        epoch: int,
        metric: Any,
        model: RegistrationNetwork,
        training_config: dict,
        optimiser: torch.optim.Optimizer | None,
    ) -> Path:
        """Save a 'last' checkpoint, growing the heap to (k+1)."""
        try:
            metric_str = f"{metric:.4f}"
        except:
            metric_str = str(metric)
        checkpoint_path = (
            self.config.root
            / self.config.checkpoints_subdir
            / f"{self._filename_prefix()}_metric{metric_str}_last.ckpt"
        )
        self._save_ckpt(
            path=checkpoint_path,
            epoch=epoch,
            metric=metric,
            model=model,
            training_config=training_config,
            optimiser=optimiser,
        )
        heapq.heappush(
            self._heap,
            _HeapCheckpoint(
                epoch=epoch,
                metric=metric,
                filepath=checkpoint_path,
                is_better=self.config.is_better,
            ),
        )
        return checkpoint_path

    def compress(self, keep_original: bool = False, mode: str = "xz"):
        """Compresses all checkpoints in the heap."""
        if mode not in (supported := ("xz", "bz2", "gz")):
            raise ValueError(f"{mode=} not in supported modes: {supported}")
        if mode == "xz":
            kwargs = {"preset": 9}
        else:
            kwargs = {"compresslevel": 9}
        for checkpoint in self.topk_descending():
            src = checkpoint.filepath
            dst = src.with_suffix(f".ckpt.tar.{mode}")
            with (
                zipfile.ZipFile(src, "r") as zin,
                tarfile.open(  # pyright: ignore[reportCallIssue]
                    dst,
                    f"w:{mode}",  # pyright: ignore[reportArgumentType]
                    **kwargs,  # pyright: ignore[reportArgumentType]
                ) as tout,
            ):
                for info in zin.infolist():
                    zi = zipfile.ZipInfo(
                        info.filename, date_time=info.date_time
                    )
                    zi.external_attr = info.external_attr
                    zi.create_system = info.create_system
                    zi.comment = info.comment
                    with zin.open(info, "r") as fsrc:
                        ti = tarfile.TarInfo(info.filename)
                        ti.size = info.file_size
                        ti.mtime = time.mktime(info.date_time + (0, 0, -1))
                        tout.addfile(ti, fsrc)
            if not keep_original:
                checkpoint.filepath = dst
                src.unlink()

    def topk_descending(self) -> list[_HeapCheckpoint]:
        """Get the top-k checkpoints' paths in descending order."""
        return sorted(
            self._heap,
            key=functools.cmp_to_key(
                lambda a, b: (
                    -1 if self.config.is_better(a.metric, b.metric) else 1
                )
            ),
        )

    def cleanup(self):
        """Deletes all checkpoints."""
        shutil.rmtree(self.config.root / self.config.checkpoints_subdir)
        self._heap.clear()
