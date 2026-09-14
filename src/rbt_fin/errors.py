# -*- coding: utf-8 -*-
"""
公共异常定义。

功能或目的：
    为整个数据层提供一组“语义明确、可被调用方分类处理”的异常。所有异常都必须传播到
    调用方（Issue #5 §6：写失败只打印不抛是被明令禁止的旧行为）。

调用关系：
    - `rbt_fin.http` 抛出 `AcquisitionError` 族（网络/HTTP/业务状态/结构错误）。
    - `rbt_fin.validation` 抛出 `ContractError` 族（时间轴/数值/覆盖/修订）。
    - `rbt_fin.storage` 抛出 `StorageError` 族（锁/发布/缓存损坏）。
    - `rbt_fin.pipeline` 抛出 `CacheEligibilityError` 族（offline 不可用/陈旧超限）。
"""

from __future__ import annotations


class RbtFinError(Exception):
    """本数据层所有异常的根类型；调用方可用它一次性区分“本层错误 vs 其他错误”。"""


# -------------------- 配置 / 入参 --------------------

class ConfigError(RbtFinError):
    """数据集定义、请求参数或运行期配置非法（例如 naive datetime、start >= end）。"""


# -------------------- 采集 --------------------

class AcquisitionError(RbtFinError):
    """采集阶段失败的根类型：网络、HTTP 状态、业务状态码或响应结构错误。"""


class HttpStatusError(AcquisitionError):
    """HTTP 层面失败（非 2xx），包含重试耗尽后的最终失败。"""


class BusinessStatusError(AcquisitionError):
    """HTTP 200 但载荷内的业务状态表示失败（例如 Bybit `retCode != 0`）。"""


class ResponseSchemaError(AcquisitionError):
    """响应结构与适配器预期不符（字段缺失、类型错误、行宽不足）。"""


class SourceBlockedError(AcquisitionError):
    """
    来源在当前运行环境被拦截（反爬页、JS 挑战、地域限制等），而不是协议漂移。

    为什么单独成类型：`ResponseSchemaError` 意味着"字段变了，去改代码"，
    而这里意味着"换出口或换来源"。两者混用会把运维引向错误的排查方向。
    """


class PaginationError(AcquisitionError):
    """分页协议异常：重复页、游标不推进、返回越界记录。"""


class BudgetExhaustedError(AcquisitionError):
    """请求数/页数预算耗尽；本次扫描未完成，禁止按“完整”对待。"""


# -------------------- 数据合同 --------------------

class ContractError(RbtFinError):
    """数据内容违反数据集声明的合同（时间轴、数值、主键、单位）。"""


class TimeContractError(ContractError):
    """时间轴违约：非 UTC、不在网格上、非严格升序、重复主键。"""


class ValueContractError(ContractError):
    """数值违约：非有限值、价格区间关系错误、成交量为负等。"""


class CoverageError(ContractError):
    """严格请求所要求的覆盖未达成（头部/内部/尾部缺失，或覆盖状态为未验证）。"""


class RevisionConflictError(ContractError):
    """同主键出现未经解释的不同内容，或历史被静默缩短。"""


# -------------------- 存储 --------------------

class StorageError(RbtFinError):
    """存储层失败的根类型。"""


class LockError(StorageError):
    """无法获取单数据集写锁（超时或被占用）。"""


class CacheCorruptionError(StorageError):
    """manifest 与内容文件不匹配，或内容文件哈希与文件名/manifest 声明不一致。"""


class PublishError(StorageError):
    """内容文件写入、manifest 写入或原子替换失败。"""


# -------------------- 缓存资格 --------------------

class CacheEligibilityError(RbtFinError):
    """缓存不满足本次请求，且当前模式不允许（或无法）联网刷新。"""


class OfflineUnavailableError(CacheEligibilityError):
    """`offline` 模式下缓存缺失或不满足请求范围/覆盖要求。"""


class StaleCacheError(CacheEligibilityError):
    """刷新失败后旧缓存已超过允许的陈旧上限，不能继续使用。"""
