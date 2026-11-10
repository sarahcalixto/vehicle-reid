from reid.data.dataset import parse_filename

def test_parse_filename():
    assert parse_filename("0025_c016_00045735_0.jpg") == (25, 16)
    assert parse_filename("0613_c008_00062940_0.jpg") == (613, 8)
    assert parse_filename("0165_c003_00075690_0.jpg") == (165, 3)