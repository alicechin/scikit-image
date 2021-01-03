from GHM import *
from GHMHelperFuncs import *

house = read_and_check_img("house.jpg")
streetlights = read_and_check_img("streetlights.jpg")

matched = cdfGHM(house, streetlights)
show_img(matched)