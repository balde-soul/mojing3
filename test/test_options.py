
def a(**options):
    if options.pop('a') == True:
        assert options.pop('b', None) is None
        pass
    pass

if __name__ == '__main__':
    a(a=True)