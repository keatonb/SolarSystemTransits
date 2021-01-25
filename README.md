# Solar System Transits

Code to calculate and plot details of inner Solar System transits as seen from outer Solar System objects.

Details in Bell & Rustamkulov (2021, in prep.).

Requires package solarsystem (https://pypi.org/project/solarsystem/).
Animation requires imagemagick (https://imagemagick.org).


Try the following to make an animated gif of Earth's next transit of the Sun from Uranus' perspective.
```python3
from SSTransits import Transit
from datetime import datetime

starttime = datetime(2024,11,16,14)
endtime = datetime(2024,11,17,16)
transit = Transit("Earth","Uranus",starttime,endtime,10)
transit.animate()
```
![Transit gif](https://github.com/keatonb/SolarSystemTransits/raw/main/Transit.gif)
