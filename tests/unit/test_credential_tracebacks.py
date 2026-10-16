"""使用真实 requests 异常，验证重试失败和预算拒绝均不向堆栈或日志泄露凭据。"""

import logging
import traceback

import pytest
import requests
import urllib3

from rbt_fin.errors import BudgetExhaustedError, HttpStatusError
from rbt_fin.http import HttpClient, RequestBudget, build_session


class _FailingAdapter(requests.adapters.BaseAdapter):
    """离线模拟连接失败；保留真实请求 URL，让测试覆盖 urllib3 携带的查询串。"""

    def __init__(self):
        """初始化发送计数，用于核对重试次数没有被安全修复改变。"""
        super().__init__()
        self.calls = 0

    def send(self, request, **kwargs):
        """输入 PreparedRequest，抛出含请求 URL 的连接错误，不访问网络。"""
        self.calls += 1
        # requests 会把底层 urllib3 错误保留在异常参数中，仅改外层文字并不安全。
        error = urllib3.exceptions.MaxRetryError(
            pool=None, url=request.path_url, reason="simulated connection refused"
        )
        raise requests.exceptions.ConnectionError(error, request=request)

    def close(self):
        """替身不持有连接，关闭时无需释放外部资源。"""


@pytest.mark.parametrize(argnames="method_name", argvalues=("get_json", "get_text"))
@pytest.mark.parametrize(
    argnames="max_retries,wait_budget,error_type,expected_calls",
    argvalues=(
        (0, 10.0, HttpStatusError, 1),
        (1, 10.0, HttpStatusError, 2),
        (1, 0.0, BudgetExhaustedError, 1),
    ),
)
def test_transport_failures_never_attach_credentials(
    method_name, max_retries, wait_budget, error_type, expected_calls, caplog
):
    """完整堆栈和 logging.exception 均不含 key，且重试/等待预算继续生效。"""
    # 使用虚构凭据和本地适配器；真实 Session 仍会构造带查询参数的请求 URL。
    secret = "fake-query-credential-for-traceback-tests"
    adapter = _FailingAdapter()
    session = build_session()
    session.mount(prefix="https://", adapter=adapter)
    client = HttpClient(
        session=session,
        budget=RequestBudget(
            max_requests=5,
            max_retries=max_retries,
            backoff_seconds=1.0,
            max_total_wait_seconds=wait_budget,
        ),
        sleep=lambda seconds: None,
    )
    request = getattr(client, method_name)
    logger = logging.getLogger(name=__name__)

    # 模拟消费者记录完整异常；预算拒绝发生在网络失败后的退避路径，也必须安全。
    with caplog.at_level(level=logging.ERROR), pytest.raises(error_type) as caught:
        try:
            request(
                url="https://example.invalid/series",
                params={"api_key": secret, "series_id": "DGS10"},
                context="credential-audit",
            )
        except error_type:
            logger.exception(msg="credential-audit: 采集失败")
            raise

    # 不仅遮住最终文字，还要求不挂接含请求信息的原始异常。
    error = caught.value
    formatted = "".join(traceback.format_exception(
        type(error), error, error.__traceback__
    ))
    assert secret not in formatted
    assert secret not in caplog.text
    assert error.__cause__ is None
    assert error.__context__ is None
    assert "credential-audit" in str(error)
    assert adapter.calls == expected_calls
    assert client.requests_used == expected_calls
    if error_type is HttpStatusError:
        assert "ConnectionError" in str(error)
        assert "api_key=<redacted>" in str(error)
