import editdistance


def err(n, m):
    dist = editdistance.eval(n, m)
    if len(n) == 0:
        return len(m)
    return dist / len(n)

def nn_err(n, m):
    dist = editdistance.eval(n, m)
    if len(n) == 0:
        return len(m)
    return dist


def cer(n, m):
    n = ' '.join(n.split())
    m = ' '.join(m.split())
    return err(n, m)


def wer(n, m):
    n = n.split()
    m = m.split()
    return err(n, m)

def nn_cer(n, m):
    n = ' '.join(n.split())
    m = ' '.join(m.split())
    return nn_err(n, m)


def nn_wer(n, m):
    n = n.split()
    m = m.split()
    return nn_err(n, m)
