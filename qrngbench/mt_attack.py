"""Why Mersenne Twister must never make keys: recover its full state from 624
consecutive 32-bit outputs and predict everything after. Educational demo
against your OWN local random.Random instance."""
import random


def _untemper(y):
    y ^= y >> 18
    y ^= (y << 15) & 0xEFC60000
    t = y
    for _ in range(4):
        t = y ^ ((t << 7) & 0x9D2C5680)
    y = t
    t = y
    for _ in range(2):
        t = y ^ (t >> 11)
    return t & 0xFFFFFFFF


def clone(outputs):
    assert len(outputs) >= 624
    state = [_untemper(o) for o in outputs[:624]]
    r = random.Random()
    r.setstate((3, tuple(state + [624]), None))
    return r


def demo(n_predict=1000):
    victim = random.Random()           # unknown seed
    seen = [victim.getrandbits(32) for _ in range(624)]
    attacker = clone(seen)
    ok = all(attacker.getrandbits(32) == victim.getrandbits(32) for _ in range(n_predict))
    return ok, n_predict
