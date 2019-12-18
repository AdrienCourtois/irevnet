def renormalize(x):
    x = x.copy()
    x = x * 0.3081 + 0.1307
    x[x < 0] = 0
    x[x > 1] = 1

    return x