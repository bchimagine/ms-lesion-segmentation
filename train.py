import os
import sys

sys.path.insert(0, os.getcwd()+'\lib')

config = {}
with open("config.txt") as file:
	for line in file:
		(key, val) = line.split(':')
		val = val.strip()
		val = " ".join(val.split())
		config[key] = val

if config['network'] == '2d-dense':
	import densenet as densenet
elif config['network'] == '3d-dense':
	import densenet3D as densenet
elif config['network'] == '3d-dense-compact':
	import densenet3DD as densenet
else:
	print("Selected Network Architecture", config['network'], "not recognized.")
	sys.exit()

