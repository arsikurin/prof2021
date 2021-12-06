import screen_brightness_control as sbc
import time
# import sys
# import functools as ft
# if sys.platform == "linux":
#     ft.partial(sbc.set_brightness, )
print(sbc.get_brightness())

sbc.set_brightness(50)

print(sbc.get_brightness())
time.sleep(3)
# set the brightness of the primary display to 75%
sbc.set_brightness(75)

print(sbc.get_brightness())

# fade the brightness from 25% to 75%
sbc.fade_brightness(finish=1, start=100)
print(sbc.get_brightness())
