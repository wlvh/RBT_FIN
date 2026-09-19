# -*- coding: utf-8 -*-
"""
包身份信息（版本与数据格式版本）。

功能或目的：
    单独成文件，避免 `rbt_fin/__init__.py` 为了拿版本号而导入重量级模块，
    也让 manifest 里的 `code_version` 有一个稳定的真相源。
"""

from __future__ import annotations

__version__ = "0.3.3"

# trading 兼容 CSV 的导出格式版本；与数据本身的 schema_version 是两个独立概念。
LEGACY_CSV_FORMAT_VERSION = "legacy-ohlcv-1"
