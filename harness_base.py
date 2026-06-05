"""
harness_base.py — Harness 基类
"""

class Harness:
    def __init__(self, call_llm):
        self.call_llm = call_llm                          # 调用 LLM
        self.memory: list[tuple[str, str]] = []

    def update(self, text: str, label: str) -> None:
        """接收一条带标签的训练样本，更新内部记忆"""
        self.memory.append((text, label))

    def predict(self, _text: str) -> str:
        """对文本预测标签，返回标签字符串"""
        raise NotImplementedError

    def name(self) -> str:
        return self.__class__.__name__
