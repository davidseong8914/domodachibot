# to reset pw
sudo passwd pi

# find ip
ip addr

# to run camera on pi
rpicam-hello -t 0



## pi and show on mac combi
sshed terminal:
```
rpicam-vid -t 0 -n --inline --codec h264 --listen -o tcp://0.0.0.0:8554
```
on mac terminal:
```
ffplay -fflags nobuffer tcp://raspberrypi.local:8554
```

## i have a gate detection dataset on roboflow
my andrew account