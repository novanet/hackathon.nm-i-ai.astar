import sys; sys.path.insert(0, '.')
from train_unet import load_round_data
d, g = load_round_data('597e60cf-d1a1-4627-ac4d-2a61da68b6df')
print('detail:', bool(d), 'gts:', len(g))
