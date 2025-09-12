import hashlib

MOD = 2 ** 32


def make_seed(*parts: str, mod: int = MOD) -> int:
    h = hashlib.sha256("||".join(map(str, parts)).encode()).hexdigest()
    return int(h[:8], 16) % mod


def seed_for_image(namespace: str, topic: str, seed_group: str, image_index: int, entity_id: str | None = None) -> int:
    base = [namespace, topic, seed_group, image_index]
    if entity_id:
        base.insert(2, entity_id)
    return make_seed(*base)


