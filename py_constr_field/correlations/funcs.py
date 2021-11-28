import os
from functools import wraps

from shove import Shove


class LRUCache:
    def __init__(self, cache_dir="cache"):
        self.cache_dir = os.path.abspath(os.path.expanduser(cache_dir))
        if not os.path.isdir(self.cache_dir):
            os.makedirs(self.cache_dir)
        self.file_store = Shove(
            cache="simple://",
            store="lite://" + os.path.join(self.cache_dir, "data.sqlite"),
        )

    def __call__(self, fun):
        name = fun.__name__

        @wraps(fun)
        def wrapper(*args):
            key = str((name, args))

            if key in self.file_store:
                val = self.file_store[key]
            else:
                val = fun(*args)
                self.file_store[key] = val

            return val

        return wrapper
