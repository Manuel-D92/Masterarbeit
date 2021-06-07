import pylab as pl
from sklearn import datasets, svm, metrics

digits = datasets.load_digits()

print ("Shape of digits.data: \t",digits.data.shape)
print ("Shape of digits.images: \t",digits.images.shape)
print ("Shape of digits.target: \t",digits.target.shape)
n_samples=digits.images.shape[0]
print ("Number of samples: \t",n_samples)
# Ausgabe der ersten 4 Images und ihres Klassenindex
for index, (image, label) in enumerate(zip(digits.images, digits.target)[:4]):
    pl.subplot(2, 4, index+1)
    pl.imshow(image, cmap=pl.cm.gray_r)
    pl.title('Training: %i' % label)