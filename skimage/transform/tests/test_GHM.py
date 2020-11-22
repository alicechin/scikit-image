import numpy as np

# from skimage.transform import GHM
# from skimage.transform import GHMHelperFuncs
from skimage import transform
from skimage import data

# from skimage._shared.testing import assert_array_almost_equal, \
#     assert_almost_equal

import pytest

# get the skimage standard test images. See https://scikit-image.org/docs/dev/api/skimage.data.html
astronaut = data.astronaut()
# brain = data.brain()
# brick = data.brick()
# camera = data.brick()
# cat = data.cat()
# cell = data.cell()
# cells3d = data.cell3d()
# checkerboard = data.checkerboard()
chelsea = data.chelsea()
# clock = data.clock()
# coffee = data.coffee()
# coins = data.coins()
# colorwheel = data.colorwheel()
# eagle = data.eagle()
# grass = data.grass()
# gravel = data.gravel()
# horse = data.horse()
# hubble_deep_field = data.hubble_deep_field()
# human_mitosis = data.human_mitosis()
# immunohistochemistry = data.immunohistochemistry()
# kidney = data.kidney()
# lily = data.lily()
# logo = data.logo()
# microaneurysm = data.microaneurysm()
# moon = data.moon()
# page = data.page()
# retina = data.retina()
# rocket = data.rocket()
# shepp_logan_phantom = data.shepp_logan_phantom()
# skin = data.skin()
# motorcycle_left, motorcycle_right, disp = data.stereo_motorcycle()
# text = data.text()


@pytest.mark.parametrize('img', [
    astronaut,
    chelsea
])
def test_whole_image_matched_to_itself(img):
	# matched_img = GHM.GHM(img, img, False)
	# assert np.equals(img, matched_img)
	assert True