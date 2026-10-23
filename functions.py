from value import Value

def test_grad():
    h = 0.001

    a = Value(2, label='a')
    b = Value(-3, label='b')
    e = a * b
    e.label = "e"
    c = Value(10, label='c')
    d = e + c
    d.label = "d"
    f = Value(-2, label='f')
    L1 = d * f
    L1.label = "L1"

    a = Value(2, label='a')
    b = Value(-3, label='b')
    e = a * b
    e.label = "e"
    c = Value(10, label='c')
    d = e + c
    d.label = "d"
    f = Value(-2, label='f')
    L2 = d * f
    L2.label = "L2"

    print("L1.data", L1.data)
    print("L2.data", L2.data)
    print((L2.data - L1.data) / h)
