import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import collections
import time
import pickle
from pathlib import Path
import os

_since_beginning = collections.defaultdict(lambda: {})
_since_last_flush = collections.defaultdict(lambda: {})


def write_to_console_file(txt, name):
	Path(os.path.dirname(name)).mkdir(parents=True, exist_ok=True)
	with open(name, 'a') as f:
		f.write(txt + '\n')
		print(txt)


_iter = [0]
def tick():
	_iter[0] += 1

def plot(name, value):
	_since_last_flush[name][_iter[0]] = value

def flush(dir,logname=None):
	prints = []

	for name, vals in _since_last_flush.items():
		prints.append("{}\t{}".format(name, np.mean(list(vals.values()))))
		_since_beginning[name].update(vals)

		x_vals = np.sort(list(_since_beginning[name].keys()))
		y_vals = [_since_beginning[name][x] for x in x_vals]

		plt.clf()
		plt.plot(x_vals, y_vals)
		plt.xlabel('iteration')
		plt.ylabel(name)
		plt.savefig(os.path.join(dir,name.replace(' ', '_')+'.jpg'))

	if logname is None:
		print("iter {}\t{}".format(_iter[0], "\t".join(prints)))
	else:
		write_to_console_file("iter {}\t{}".format(_iter[0], "\t".join(prints)), os.path.join(dir, logname))

	_since_last_flush.clear()

	with open(os.path.join(dir, 'log.pkl'), 'wb') as f:
		pickle.dump(dict(_since_beginning), f, pickle.HIGHEST_PROTOCOL)
