def merge(x, d):
    # x - input, d - class width
    res = x[0]
    buf = []
    for i in range(1, len(x)):
        if len(buf) == 0:
            buf = [res[- 1]]
        if abs(buf[-1] - x[i]) < d:
            buf.append(x[i])
        else:
            res.append(buf[-1])
            buf = []
    print(res)
