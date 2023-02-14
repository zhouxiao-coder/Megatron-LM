class BestOfNModel:
    def __init__(self, n: int, pre_process=True, post_process=True):
        raise NotImplementedError("""基本上应该和Classification 模型一致。

BestOfN Model的输入是一个 text，输出是一个 logit

而List-wise的softmax，在loss的时候进行处理。
""")
