def merge(x, ind, d):
    # x - input, d - class width
    res = [x[0]]
    buf = []
    for i in range(1, len(x)):
        if len(buf) == 0:
            buf.append(res[- 1])

        if abs(buf[-1][ind] - x[i][ind]) <= d:
            buf.append(x[i])

        else:
            if len(buf) > 1:
                res.append(buf[-1])
            else:
                res.append(x[i])
            buf = []
            
        if i == (len(x) - 1):
            res.append(x[i])

    return res