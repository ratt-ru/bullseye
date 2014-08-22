import time

class timer(object):
    def __init__(self):
      self.reset()
    def __enter__(self):
      self.start()
      return self
    def __exit__(self, *args):
      self.stop()
    def start(self):
      self._start = time.time()
    def stop(self):
      end = time.time()
      self._secs += end - self._start
    def elapsed(self):
      return self._secs
    def reset(self):
      self._secs = 0