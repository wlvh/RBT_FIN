# -*- coding: utf-8 -*-
"""
来源适配器包。

功能或目的：
    每个模块负责一个端点族的“参数语义 + 分页边界 + 空结果语义 + 业务错误判定”。
    公共下载可靠性（超时/重试/预算）在 `rbt_fin.http`；公共校验在 `rbt_fin.validation`。
    分页边界必须由适配器自己负责，因为不同端点的 startTime/endTime/游标语义无法互相推导
    （Issue #5 §4.1、§5）。

调用关系：
    `rbt_fin.registry` 把这里的 `acquire_*` 函数绑定到具体 `Dataset`；
    `rbt_fin.pipeline` 通过 `Dataset.acquire(context)` 调用它们。
"""
