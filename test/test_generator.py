
def generator(size, type):
    a = list(range(0, size))
    if type == 0:
        for i in range(0, int(size / 2)):
            yield a[i]
    else:
        for i in range(int(size / 2), size):
            yield a[i]



if __name__ == '__main__':
    gen = generator(100, 0)
    gen1 = generator(100, 1)
    for i in range(0, 55):
        try:
            a = gen.__next__()
        except Exception:
            gen = generator(100, 0)
            a = gen.__next__()
        print(a)
    print("======")
    for i in range(50, 100):
        print(gen1.__next__())