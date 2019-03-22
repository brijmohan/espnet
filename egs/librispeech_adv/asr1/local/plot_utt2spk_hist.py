import numpy as np

# matplotlib related
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

savfile = 'local/train_960_utt2spk.svg'

utt2spk = 'data/train_960_org/spk2utt'
with open(utt2spk) as f:
    lines = f.read().splitlines()
    x = [len(l.split()[1:]) for l in lines]

col = ['g'] * len(x)
for i, v in enumerate(x):
    if v < 50:
        col[i] = 'r'
    elif v < 100 and v > 50:
        col[i] = 'y'
    elif v < 150 and v > 100:
        col[i] = 'b'

h = np.array(x)
x = np.arange(len(x))

print "< 50 --> ", len([c for c in col if c == 'r'])
print "50 - 100 --> ", len([c for c in col if c == 'y'])
print "100 - 150 --> ", len([c for c in col if c == 'b'])

# the histogram of the data
plt.bar(x, h, color=col)

plt.xlabel('Speakers')
plt.ylabel('#utt')
plt.title('bar chart of speaker utterances')
#plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
#plt.axis([40, 160, 0, 0.03])
plt.grid(True)
plt.savefig(savfile, dpi=300)

