
class Value:
    def __init__(self, data: float, _children: tuple = (), _op: str = '', label: str = ''):
        self.data = data
        self._prev = set(_children)
        self._op = _op
        self.label = label
        self.grad = 0

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other: Value):
        return Value(self.data + other.data, (self, other), '+')
    
    def __mul__(self, other: Value):
        return Value(self.data * other.data, (self, other), '*')
