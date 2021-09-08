

class Logger:
    def __init__(self, path):
        self._path = path

    def log(self, s):
        with open(self._path, 'a') as f:
            f.write(str(s) + '\n')
