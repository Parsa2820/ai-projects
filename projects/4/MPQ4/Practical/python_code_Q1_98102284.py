student_number = 98102284
Name = 'Parsa'
Last_Name = 'Mohammadian'


from Helper_codes.validator import *

python_code = extract_python("./Q1.ipynb")
with open(f'python_code_Q1_{student_number}.py', 'w') as file:
    file.write(python_code)


!pip install torchvision
!pip install numpy

from torchvision import datasets
import numpy as np
from scipy.stats import multivariate_normal


train_data = datasets.MNIST('./data', train=True, download=True)
test_data = datasets.MNIST('./data', train=False, download=True)

train_images = np.array(train_data.data)
train_labels = np.array(train_data.targets)
test_images = np.array(test_data.data)
test_labels = np.array(test_data.targets)


class Bayes:
    def train(self, train_images, train_lables):
        dim = 784    # 28 * 28
        self.gaussian = {}
        self.label_pros = {}
        self.labels = set(train_labels)
        smoothing = None
        smoothing = 4000
        for label in self.labels:
            data = train_images[train_lables == label].reshape(-1, dim)
            mean = data.mean(axis=0)
            cov = np.cov(data, rowvar=False) + np.eye(dim) * smoothing
            self.gaussian[label] = multivariate_normal(mean, cov)
            self.label_pros[label] = len(
                train_lables[train_lables == label]) / len(train_lables)

    def calc_accuracy(self, images, labels):
        return np.mean(self.predict_labels(images) == labels)

    def predict_labels(self, images):
        return [self.predict_label(image.reshape(-1)) for image in images]

    def predict_label(self, image):
        return max(self.labels, key=lambda l: self.calculate_log_prob(image, l).mean())

    def calculate_log_prob(self, image, label):
        return np.log(self.label_pros[label]) + self.gaussian[label].logpdf(image)


network = Bayes()
network.train(train_images, train_labels)


print("Accuracy on test data (%) : " + str(network.calc_accuracy(test_images[:100], test_labels[:100]) * 100))

