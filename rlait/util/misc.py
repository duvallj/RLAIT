class dotdict(dict):
    def __getattr__(self, name):
        if name.beginswith('__'): return super().__getattr__(name)
        return self[name]
