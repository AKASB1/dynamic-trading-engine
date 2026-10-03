"""Stream derivation (QC 1.2) against the known answers of QC Appendix B."""

from dynamic_trading_engine.rng import fnv1a64, splitmix64, stream, stream_seeds


def test_fnv1a64_known_answers():
    assert fnv1a64("") == 0xCBF29CE484222325
    assert fnv1a64("a") == 0xAF63DC4C8601EC8C
    assert fnv1a64("foobar") == 0x85944171F73967E8


def test_splitmix64_known_answers():
    assert splitmix64(0) == 0xE220A8397B1DCDAF
    assert splitmix64(1) == 0x910A2DEC89025CC1


def test_stream_known_answers():
    s1, s2 = stream_seeds(42, "alpha")
    assert s1 == 0xDDA774F898BBCFB5
    assert s2 == 0xE8845643B324C5EA
    g = stream(42, "alpha")
    got = [g.random() for _ in range(3)]
    want = [0.16959585488328055, 0.282582992635685, 0.6960747057340525]
    for a, b in zip(got, want):
        assert abs(a - b) <= 1e-9 * abs(b)


def test_streams_are_independent_of_each_other():
    a1 = stream(7, "x").random(5)
    stream(7, "y").random(100)  # drawing from another component changes nothing
    a2 = stream(7, "x").random(5)
    assert (a1 == a2).all()


def test_normal_prefix_property():
    # the forecast noise relies on it: the first k draws do not depend on how many are drawn
    a = stream(3, "forecast.noise.I001").standard_normal(300)
    b = stream(3, "forecast.noise.I001").standard_normal(1000)
    assert (a == b[:300]).all()
