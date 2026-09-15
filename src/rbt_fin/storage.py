# -*- coding: utf-8 -*-
"""
内容寻址的不可变缓存 + manifest 原子提交。

功能或目的：
    实现 Issue #5 §6 的最小可靠存储合同：

        cache/<dataset>/
          data-<sha256>.csv     内容文件，写入后永不原地修改
          manifest.json         唯一提交点，原子替换
          update.lock           单数据集单写者的固定旁路锁

    发布流程：取锁 -> 读旧 manifest -> 独立临时文件构造/校验 -> 写内容文件 ->
    校验哈希与可读性并 fsync -> 写临时 manifest -> 同文件系统 os.replace -> 目录 fsync -> 返回。

调用关系：
    - `rbt_fin.pipeline` 通过 `DatasetStore.lock()` 包住“读旧状态 -> 更新 -> 校验 -> 发布”。
    - `DatasetStore.publish()` 是唯一写入口；`read_manifest()/read_frame()` 是唯一读入口。
    - `phase_hook` 用于在受控测试中于提交前/提交后精确中断进程（Issue #5 §8 两个关键中断）。
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Optional, Sequence

import pandas as pd

from .errors import CacheCorruptionError, ConfigError, LockError, PublishError
from .timeutils import require_utc

MANIFEST_FILENAME = "manifest.json"
LOCK_FILENAME = "update.lock"
CONTENT_PREFIX = "data-"
CONTENT_SUFFIX = ".csv"

# manifest 自身的格式版本；与数据集的 schema_version 是两个独立概念。
MANIFEST_FORMAT_VERSION = 1

# 发布流程中的命名同步点。测试通过 phase_hook 在这些点注入中断/等待，
# 从而区分“提交前中断”（旧状态必须不变）与“提交后中断”（新状态可能已可见）。
PHASE_BEFORE_CONTENT_WRITE = "before_content_write"
PHASE_AFTER_CONTENT_WRITE = "after_content_write"
PHASE_BEFORE_MANIFEST_REPLACE = "before_manifest_replace"
PHASE_AFTER_MANIFEST_REPLACE = "after_manifest_replace"


def sha256_bytes(payload: bytes) -> str:
    """计算字节串的 sha256 十六进制摘要。"""
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path, *, chunk_size: int = 1 << 20) -> str:
    """
    流式计算文件内容的 sha256。

    Args:
        path: 文件路径。
        chunk_size: 读取块大小，默认 1MiB，避免大分钟数据一次性读入内存。

    Returns:
        十六进制摘要。
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frame_to_csv_bytes(frame: pd.DataFrame, *, columns: Sequence[str], time_column: str) -> bytes:
    """
    把数据帧序列化为规范 CSV 字节串（内容哈希的唯一来源）。

    为什么固定这些参数：
        内容寻址要求“同样的数据 -> 同样的字节”。行终止符、列序、索引处理任何一处放任
        默认值，都会让哈希在不同平台/调用路径下漂移。时间统一写成 tz-aware ISO 文本，
        legacy 的无时区格式只在导出边界转换（并有它自己的哈希）。

    Args:
        frame: 索引为 tz-aware UTC 时间、名为 `time_column` 的数据帧。
        columns: 落盘列序（含时间列）。
        time_column: 时间列名。

    Returns:
        UTF-8 编码的 CSV 字节串。
    """
    if frame.index.name != time_column:
        raise ConfigError(f"序列化要求索引名为 {time_column!r}，实际为 {frame.index.name!r}")
    flat = frame.reset_index()
    missing = [col for col in columns if col not in flat.columns]
    if missing:
        raise ConfigError(f"序列化缺少列: {missing}；实际列: {list(flat.columns)}")
    text = flat[list(columns)].to_csv(index=False, lineterminator="\n")
    return text.encode("utf-8")


def _finalize_csv_frame(
    frame: pd.DataFrame, *, columns: Sequence[str], time_column: str
) -> pd.DataFrame:
    """
    校验列序并把时间列还原为 tz-aware UTC 索引。

    Raises:
        CacheCorruptionError: 列集合与期望不符（缓存被外部改写或 schema 漂移）。
    """
    if list(frame.columns) != list(columns):
        raise CacheCorruptionError(
            f"内容文件列序与期望不符；期望 {list(columns)}，实际 {list(frame.columns)}"
        )
    frame[time_column] = pd.to_datetime(frame[time_column], utc=True, format="ISO8601")
    return frame.set_index(time_column)


def csv_bytes_to_frame(payload: bytes, *, columns: Sequence[str], time_column: str) -> pd.DataFrame:
    """
    `frame_to_csv_bytes` 的逆操作（内存版），用于小数据与测试。

    Args:
        payload: CSV 字节串。
        columns: 期望列序（含时间列）。
        time_column: 时间列名。

    Returns:
        索引为 tz-aware UTC 时间的数据帧。
    """
    return _finalize_csv_frame(
        pd.read_csv(BytesIO(payload)), columns=columns, time_column=time_column
    )


def csv_path_to_frame(path: Path, *, columns: Sequence[str], time_column: str) -> pd.DataFrame:
    """
    直接从文件路径解析内容文件（流式版）。

    为什么单独提供：
        生产规模的分钟数据落盘后有几百 MB。先 `read_bytes()` 再交给 `read_csv` 会同时
        持有“整份字节”和“解析后的 DataFrame”，峰值内存凭空翻倍。读取与回读校验都走
        这个入口，只有测试与小载荷才用内存版。

    Args:
        path: 内容文件路径。
        columns: 期望列序（含时间列）。
        time_column: 时间列名。

    Returns:
        索引为 tz-aware UTC 时间的数据帧。
    """
    return _finalize_csv_frame(pd.read_csv(path), columns=columns, time_column=time_column)


@dataclass(frozen=True)
class Manifest:
    """
    当前正式状态的说明文件与内容指向（manifest 本身即“CURRENT”，不另设第二个文件）。

    Attributes:
        manifest_format_version: manifest 结构版本。
        dataset_id: 数据集标识。
        identity: 数据身份（交易所/市场/交易对/价格类型/频率/单位/原始或转换含义）。
        params: 有效脱敏参数；凭据值永不进入。
        content_file: 内容文件名（相对 dataset 目录）。
        content_sha256: 内容文件字节哈希。
        content_bytes: 内容文件字节数，读取时作为廉价的一致性前置检查。
        rows: 行数。
        columns: 落盘列序。
        time_column / time_meaning: 主键时间列与其业务含义。
        schema_version / transform_version: 数据格式与转换口径版本。
        actual_start / actual_end: 数据实际范围（ISO-8601 UTC 文本，空数据为 None）。
        coverage: 覆盖检查片段（检查范围 + 状态 + 缺口计数）。
                  后续读取不能把旧检查范围当作新请求已通过。
        fetched_at: 本内容对应的最近一次成功获取时间。
        last_verified_at: 最近一次真实来源结果成功解析并按合同核验的时间。
        code_version: 生成该状态的代码版本。
        point_in_time_verified: 是否证明了“历史当时可得”。
        previous_content_file: 上一份内容文件名，用于保留必要的上一份（不承诺永久历史）。
        note: 可读备注。
    """

    manifest_format_version: int
    dataset_id: str
    identity: Mapping[str, Any]
    params: Mapping[str, Any]
    content_file: str
    content_sha256: str
    content_bytes: int
    rows: int
    columns: Sequence[str]
    time_column: str
    time_meaning: str
    schema_version: str
    transform_version: str
    actual_start: Optional[str]
    actual_end: Optional[str]
    coverage: Mapping[str, Any]
    fetched_at: str
    last_verified_at: str
    code_version: str
    point_in_time_verified: bool
    previous_content_file: Optional[str] = None
    note: str = ""
    extra: Mapping[str, Any] = field(default_factory=dict)

    def to_json_bytes(self) -> bytes:
        """序列化为稳定排序的 UTF-8 JSON 字节串（便于 diff 与哈希）。"""
        payload = {
            "manifest_format_version": self.manifest_format_version,
            "dataset_id": self.dataset_id,
            "identity": dict(self.identity),
            "params": dict(self.params),
            "content_file": self.content_file,
            "content_sha256": self.content_sha256,
            "content_bytes": int(self.content_bytes),
            "rows": int(self.rows),
            "columns": list(self.columns),
            "time_column": self.time_column,
            "time_meaning": self.time_meaning,
            "schema_version": self.schema_version,
            "transform_version": self.transform_version,
            "actual_start": self.actual_start,
            "actual_end": self.actual_end,
            "coverage": dict(self.coverage),
            "fetched_at": self.fetched_at,
            "last_verified_at": self.last_verified_at,
            "code_version": self.code_version,
            "point_in_time_verified": bool(self.point_in_time_verified),
            "previous_content_file": self.previous_content_file,
            "note": self.note,
            "extra": dict(self.extra),
        }
        return json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8")

    @classmethod
    def from_json_bytes(cls, payload: bytes, *, context: str) -> "Manifest":
        """
        从 JSON 字节串还原 manifest。

        Args:
            payload: manifest 文件字节。
            context: 报错上下文。

        Returns:
            Manifest 实例。

        Raises:
            CacheCorruptionError: JSON 非法或缺少必需字段。
        """
        try:
            data = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CacheCorruptionError(f"{context}: manifest 不是合法 JSON。") from exc
        if not isinstance(data, dict):
            raise CacheCorruptionError(f"{context}: manifest 顶层必须是对象。")
        required = (
            "manifest_format_version", "dataset_id", "identity", "params", "content_file",
            "content_sha256", "content_bytes", "rows", "columns", "time_column", "time_meaning",
            "schema_version", "transform_version", "actual_start", "actual_end", "coverage",
            "fetched_at", "last_verified_at", "code_version", "point_in_time_verified",
        )
        missing = [key for key in required if key not in data]
        if missing:
            raise CacheCorruptionError(f"{context}: manifest 缺少必需字段 {missing}。")
        if data["manifest_format_version"] != MANIFEST_FORMAT_VERSION:
            raise CacheCorruptionError(
                f"{context}: manifest_format_version={data['manifest_format_version']} "
                f"与当前实现 {MANIFEST_FORMAT_VERSION} 不符，拒绝按旧格式猜测。"
            )
        return cls(
            manifest_format_version=data["manifest_format_version"],
            dataset_id=data["dataset_id"],
            identity=data["identity"],
            params=data["params"],
            content_file=data["content_file"],
            content_sha256=data["content_sha256"],
            content_bytes=data["content_bytes"],
            rows=data["rows"],
            columns=data["columns"],
            time_column=data["time_column"],
            time_meaning=data["time_meaning"],
            schema_version=data["schema_version"],
            transform_version=data["transform_version"],
            actual_start=data["actual_start"],
            actual_end=data["actual_end"],
            coverage=data["coverage"],
            fetched_at=data["fetched_at"],
            last_verified_at=data["last_verified_at"],
            code_version=data["code_version"],
            point_in_time_verified=data["point_in_time_verified"],
            previous_content_file=data.get("previous_content_file"),
            note=data.get("note", ""),
            extra=data.get("extra", {}),
        )


def _fsync_directory(path: Path) -> None:
    """
    对目录执行 fsync，确保 rename/create 的目录项本身落盘。

    为什么单独做：
        `os.replace` 的原子性保证的是“替换要么发生要么不发生”，不等于目录项已持久化。
        提交后是否已确认持久，是本库必须如实报告的状态（Issue #5 §6）。
    """
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


@contextmanager
def exclusive_lock(*, lock_path: Path, context: str, blocking: bool = True) -> Iterator[None]:
    """
    以旁路锁文件取得独占写权，覆盖调用方整个“读旧状态 -> 更新 -> 校验 -> 发布”周期。

    为什么锁必须覆盖整个周期而不是只包住最后一次写：
        只给写文件加锁，后到的写者仍可能拿着更旧的输入把先到者的结果盖掉——
        `os.replace` 只保证单次替换是原子的，不提供更新过程的互斥。

    为什么用旁路文件而不是锁正式文件：
        正式文件会被 `os.replace` 换掉 inode，锁在旧 inode 上对新写者不可见。

    Args:
        lock_path: 锁文件路径；父目录不存在时会被创建。
        context: 报错上下文（数据集 / 消费者标识）。
        blocking: True 阻塞等待；False 抢不到立即抛 LockError。

    Raises:
        LockError: 非阻塞模式下锁被占用。
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    flags = fcntl.LOCK_EX if blocking else (fcntl.LOCK_EX | fcntl.LOCK_NB)
    handle = lock_path.open("a+")
    try:
        try:
            fcntl.flock(handle.fileno(), flags)
        except OSError as exc:
            raise LockError(f"{context}: 无法获取更新锁 {lock_path}: {exc}") from exc
        yield
    finally:
        # 为什么显式 unlock 再 close：close 虽会释放 flock，但显式释放让顺序在测试中可控。
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def atomic_write_bytes(
    *,
    target: Path,
    payload: bytes,
    tmp_prefix: str,
    tmp_suffix: str,
    context: str,
    fsync: bool = True,
    verify_written_hash: bool = False,
    before_replace: Optional[Callable[[], None]] = None,
    after_replace: Optional[Callable[[], None]] = None,
) -> None:
    """
    用**独立命名**的临时文件写入字节，再原子替换到目标路径。

    为什么临时文件名必须独立：
        共用固定 `.tmp` 会让两个写者打开同一个文件——后开的那个以 "w" 直接截断先开者
        已写好的内容，先开者随后把这个空文件 `replace` 成正式文件，**全程不抛异常**。
        `tempfile.mkstemp` 在同目录生成唯一名字，同时保证 `os.replace` 仍在同一文件系统内。

    为什么 fsync 必须在 replace 之前：
        `replace` 只保证“读者看到的要么是旧文件要么是新文件”，不保证新文件内容已落盘。

    Args:
        target: 目标路径。
        payload: 要写入的字节。
        tmp_prefix / tmp_suffix: 临时文件名前后缀；建议以 '.' 开头，避免被目录扫描当成正式产物。
        context: 报错上下文。
        fsync: 是否 fsync 文件与目录；False 只保证可见性、不保证持久性。
        verify_written_hash: True 时回读临时文件校验哈希，确认磁盘写入可信。
        before_replace / after_replace: 提交点前后的同步回调，供受控中断测试精确注入。

    Raises:
        PublishError: `verify_written_hash` 开启且回读哈希与内存字节不符。
        OSError: 写入或替换失败；失败时清理临时文件并原样上抛。
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(target.parent), prefix=tmp_prefix, suffix=tmp_suffix)
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            if fsync:
                os.fsync(handle.fileno())
        if verify_written_hash and sha256_file(tmp_path) != sha256_bytes(payload):
            raise PublishError(f"{context}: 临时文件哈希与内存字节不符，磁盘写入不可信。")
        if before_replace is not None:
            before_replace()
        os.replace(str(tmp_path), str(target))
    except BaseException:
        # 提交前失败：旧正式状态不变；首次初始化也不留下冒充正式版本的文件。
        if tmp_path.exists():
            tmp_path.unlink()
        raise
    if after_replace is not None:
        after_replace()
    if fsync:
        _fsync_directory(target.parent)


class DatasetStore:
    """
    单个数据集的缓存目录读写入口。

    单数据集单写者：所有写入必须在 `lock()` 内进行；第二写者拿到锁后必须重新读取当前状态。
    """

    def __init__(self, *, root: Path, dataset_id: str) -> None:
        """
        Args:
            root: 缓存根目录（例如 `<workdir>/cache`）；由调用方显式传入，不扫描 cwd。
            dataset_id: 数据集标识，同时是子目录名。
        """
        if not isinstance(dataset_id, str) or not dataset_id.strip():
            raise ConfigError("dataset_id 必须是非空字符串。")
        if "/" in dataset_id or dataset_id in {".", ".."}:
            raise ConfigError(f"dataset_id={dataset_id!r} 不能包含路径分隔符。")
        self.root = Path(root)
        self.dataset_id = dataset_id
        self.dataset_dir = self.root / dataset_id

    @property
    def manifest_path(self) -> Path:
        """当前正式状态指针路径。"""
        return self.dataset_dir / MANIFEST_FILENAME

    @property
    def lock_path(self) -> Path:
        """固定旁路更新锁路径（本身不是提交点）。"""
        return self.dataset_dir / LOCK_FILENAME

    def content_path(self, content_file: str) -> Path:
        """内容文件的绝对路径。"""
        return self.dataset_dir / content_file

    @contextmanager
    def lock(self, *, blocking: bool = True) -> Iterator[None]:
        """
        获取单数据集写锁，覆盖“读旧状态 -> 更新 -> 校验 -> 发布”全过程。

        Args:
            blocking: True 时阻塞等待；False 时抢不到立即抛 LockError。

        Raises:
            LockError: 非阻塞模式下锁被占用。
        """
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        with exclusive_lock(lock_path=self.lock_path, context=self.dataset_id, blocking=blocking):
            yield

    def read_manifest(self) -> Optional[Manifest]:
        """
        读取当前 manifest；不存在时返回 None（首次初始化）。

        Returns:
            Manifest 或 None。

        Raises:
            CacheCorruptionError: manifest 存在但无法解析。
        """
        if not self.manifest_path.exists():
            return None
        payload = self.manifest_path.read_bytes()
        return Manifest.from_json_bytes(payload, context=f"{self.dataset_id}/{MANIFEST_FILENAME}")

    def read_frame(self, manifest: Manifest, *, verify_content: bool = True) -> pd.DataFrame:
        """
        按 manifest 指向读取内容文件。

        为什么默认做完整哈希校验：
            §6 明确“缓存损坏须拒绝，不能继续按文件名信任”。文件名里的哈希只是命名约定，
            外部改写不会改名。大文件的一次 sha256 相对一次批跑成本可忽略；确需跳过时
            必须由调用方显式传 `verify_content=False` 并自行承担风险。

        Args:
            manifest: 已读取的 manifest。
            verify_content: 是否重新计算内容哈希。

        Returns:
            索引为 tz-aware UTC 时间的数据帧。

        Raises:
            CacheCorruptionError: 文件缺失、大小/哈希/列序/行数与 manifest 不符。
        """
        path = self.content_path(manifest.content_file)
        if not path.exists():
            raise CacheCorruptionError(
                f"{self.dataset_id}: manifest 指向的内容文件缺失: {path}"
            )
        actual_bytes = path.stat().st_size
        if actual_bytes != int(manifest.content_bytes):
            raise CacheCorruptionError(
                f"{self.dataset_id}: 内容文件大小与 manifest 不符（{actual_bytes} != {manifest.content_bytes}）。"
            )
        if verify_content:
            actual_hash = sha256_file(path)
            if actual_hash != manifest.content_sha256:
                raise CacheCorruptionError(
                    f"{self.dataset_id}: 内容文件哈希与 manifest 不符；"
                    f"manifest={manifest.content_sha256}, actual={actual_hash}。"
                )
        frame = csv_path_to_frame(
            path, columns=manifest.columns, time_column=manifest.time_column
        )
        if len(frame) != int(manifest.rows):
            raise CacheCorruptionError(
                f"{self.dataset_id}: 内容行数与 manifest 不符（{len(frame)} != {manifest.rows}）。"
            )
        return frame

    def publish(
        self,
        *,
        frame: pd.DataFrame,
        columns: Sequence[str],
        time_column: str,
        manifest_fields: Mapping[str, Any],
        previous: Optional[Manifest],
        phase_hook: Optional[Callable[[str], None]] = None,
        fsync: bool = True,
    ) -> Manifest:
        """
        写入内容文件并原子替换 manifest（唯一提交点）。

        Args:
            frame: 待发布数据；索引为 tz-aware UTC 时间且名为 `time_column`。
            columns: 落盘列序（含时间列）。
            time_column: 时间列名。
            manifest_fields: manifest 的业务字段（identity/params/版本/范围/覆盖/时间等）。
            previous: 上一份 manifest；用于记录 `previous_content_file` 与内容复用判定。
            phase_hook: 命名同步点回调，用于受控中断测试。
            fsync: 是否执行 fsync/目录同步；仅在明确不需要持久性保证的测试里关掉。

        Returns:
            新的 Manifest。

        Raises:
            PublishError: 内容写入、回读校验或 manifest 替换失败。
            CacheCorruptionError: 已存在同名内容文件但字节哈希不一致。
        """
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        payload = frame_to_csv_bytes(frame, columns=columns, time_column=time_column)
        content_sha = sha256_bytes(payload)
        content_file = f"{CONTENT_PREFIX}{content_sha}{CONTENT_SUFFIX}"
        target = self.content_path(content_file)

        if phase_hook is not None:
            phase_hook(PHASE_BEFORE_CONTENT_WRITE)

        if target.exists():
            # 已存在同哈希文件：先核验字节，绝不原地覆盖。
            existing_hash = sha256_file(target)
            if existing_hash != content_sha:
                raise CacheCorruptionError(
                    f"{self.dataset_id}: 已存在内容文件 {content_file} 但字节哈希为 {existing_hash}，"
                    "缓存已损坏，拒绝按文件名信任。"
                )
        else:
            self._write_content_atomically(payload=payload, target=target, fsync=fsync)

        # 回读校验：证明落盘字节可按当前 schema 正确解析，而不是只信任内存中的对象。
        readback = csv_path_to_frame(target, columns=columns, time_column=time_column)
        readback_rows = len(readback)
        del readback  # 回读只为证明可解析；立刻释放，避免与原始 frame 同时驻留内存
        if readback_rows != len(frame):
            raise PublishError(
                f"{self.dataset_id}: 内容文件回读行数不符（{readback_rows} != {len(frame)}）。"
            )

        if phase_hook is not None:
            phase_hook(PHASE_AFTER_CONTENT_WRITE)

        manifest = Manifest(
            manifest_format_version=MANIFEST_FORMAT_VERSION,
            dataset_id=self.dataset_id,
            content_file=content_file,
            content_sha256=content_sha,
            content_bytes=len(payload),
            rows=int(len(frame)),
            columns=list(columns),
            time_column=time_column,
            previous_content_file=None if previous is None else previous.content_file,
            **dict(manifest_fields),
        )
        self._replace_manifest(manifest=manifest, phase_hook=phase_hook, fsync=fsync)
        return manifest

    def _write_content_atomically(self, *, payload: bytes, target: Path, fsync: bool) -> None:
        """
        以独立命名的临时文件写入内容，再原子落到内容路径。

        为什么临时文件名必须独立：
            共用固定 `.tmp` 会让两个写者互相踩，且中断后残留文件难以与正式文件区分。
            `tempfile.mkstemp` 在同目录生成唯一名字，保证 `os.replace` 仍在同一文件系统内。
        """
        atomic_write_bytes(
            target=target,
            payload=payload,
            tmp_prefix=f".{CONTENT_PREFIX}tmp-",
            tmp_suffix=CONTENT_SUFFIX,
            context=self.dataset_id,
            fsync=fsync,
            verify_written_hash=True,
        )

    def _replace_manifest(
        self,
        *,
        manifest: Manifest,
        phase_hook: Optional[Callable[[str], None]],
        fsync: bool,
    ) -> None:
        """
        写临时 manifest 并原子替换正式 manifest。

        提交语义：
            `os.replace` 之前的任何失败都不改变旧正式状态；替换之后新状态即可见。
            替换后的 fsync 只决定“持久性是否已确认”，不改变可见性。
        """
        atomic_write_bytes(
            target=self.manifest_path,
            payload=manifest.to_json_bytes(),
            tmp_prefix=".manifest-tmp-",
            tmp_suffix=".json",
            context=self.dataset_id,
            fsync=fsync,
            before_replace=None if phase_hook is None else (lambda: phase_hook(PHASE_BEFORE_MANIFEST_REPLACE)),
            after_replace=None if phase_hook is None else (lambda: phase_hook(PHASE_AFTER_MANIFEST_REPLACE)),
        )
