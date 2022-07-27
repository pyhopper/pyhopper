import pickle

import hashlib


class EvaluationCache:
    def __init__(self):
        self._cache = {}
        self._staging = {}
        self._enabled = True

    def set_enable(self, enable):
        self._enabled = enable

    def _deep_hash(self, item):
        serialized = pickle.dumps(item)
        shake_128 = hashlib.shake_128()
        shake_128.update(serialized)
        return shake_128.digest(128)

    def __contains__(self, item):
        if not self._enabled:
            return False
        hash_code = self._deep_hash(item)
        return hash_code in self._cache or hash_code in self._staging

    # def insert(self, item, result):
    #     hash_code = self._deep_hash(item)
    #     if hash_code in self._cache:
    #         print("WARNING: Hash collision")
    #     self._cache[hash_code] = result

    def forget(self, item):
        hash_code = self._deep_hash(item)
        if hash_code in self._cache:
            del self._cache[hash_code]
        if hash_code in self._staging:
            del self._staging[hash_code]

    def stage(self, item):
        if self._enabled:
            hash_code = self._deep_hash(item)
            self._staging[hash_code] = 1

    def commit(self, item, result):
        if self._enabled:
            hash_code = self._deep_hash(item)
            if hash_code in self._staging.keys():
                del self._staging[hash_code]
            else:
                print("WARNING: hash_code not found in staging!\n" + str(item))
            self._cache[hash_code] = result

    def clear(self):
        self._cache = {}

    def state_dict(self):
        return {"cache": self._cache}

    def load_state_dict(self, state):
        self._cache = state["cache"]