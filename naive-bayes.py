import numpy as np


class NaiveBayes:

  def fit(self, X, y):
    sampleCount, featuresCount = X.shape
    self.classes = np.unique(y)
    classesCount = len(self.classes)

    self.meansOfEachClass = np.zeros((classesCount, featuresCount), dtype=np.float64)
    self.varianceOfEachClass = np.zeros((classesCount, featuresCount), dtype=np.float64)
    self.priorPropabilities = np.zeros(classesCount, dtype=np.float64)

    for idx, c in enumerate(self.classes):
      X_c = X[y == c]
      self.meansOfEachClass[idx, :] = X_c.mean(axis=0)
      self.varianceOfEachClass[idx, :] = X_c.var(axis=0)
      self.priorPropabilities[idx] = X_c.shape[0] / float(sampleCount)

  def predict(self, X):
    yPred = [self._predcitEach(x) for x in X]
    return np.array(yPred)

  def _predcitEach(self, x):
    posteriors = []

    for idx, c in enumerate(self.classes):
      prior = np.log(self.priorPropabilities[idx])
      posterior = np.sum(np.log(self.getPDF(idx, x)))
      posterior = prior + posterior
      posteriors.append(posterior)

    return self.classes[np.argmax(posteriors)]

  def getPDF(self, classIdx, x):
    mean = self.meansOfEachClass[classIdx]
    variance = self.varianceOfEachClass[classIdx]
    numerator = np.exp(- (x-mean)**2 / (2 * variance))
    denominator = np.sqrt(2 * np.pi * variance)
    return numerator / denominator


class LabelEncoder:

  def fitTransform(self, X):
    data = []
    self.map = {}
    self.labelCount = 0

    for each in X:
      if each in self.map:
        data.append(self.map[each])
      else:
        self.map[each] = self.labelCount
        self.labelCount += 1
        data.append(self.map[each])

    return data

  def inverseTransform(self, X):
    data = []
    if not self.map:
      return None

    for each in X:
      for value, key in enumerate(self.map):
        if each == value:
          data.append(key)

    return data


def accuracy(yTrue, yPred):
    accuracy = np.sum(yTrue == yPred) / len(yTrue)
    return accuracy


attributes = []
m_class = []


if __name__ == "__main__":
  with open('dataset/iris.data') as file:
    data = file.read()
    lines = data.split('\n')
    for line in lines:
      row = line.split(',')
      if len(row) > 1:
        features = list(map(float, row[:-1]))
        label = row[-1]
        attributes.append(features)
        m_class.append(label)

  le = LabelEncoder()
  size = int(len(attributes) - (len(attributes) * .2))
  classes = le.fitTransform(m_class)
  print("Spliting dataset in to 80% training data, 20% testing data")
  y_train = np.array(classes[:size])
  X_train = np.array(attributes[:size], np.float64)
  X_test = np.array(attributes[size:], np.float64)
  y_test = np.array(classes[size:])

  model = NaiveBayes()
  print("Training model...")
  model.fit(X_train, y_train)
  print("Testing...")
  y_pred = model.predict(X_test)
  y_pred_labeled = le.inverseTransform(y_pred)

  print(f"Accuracy: {accuracy(y_test, y_pred)*100:.2f}%, try:")
  a = float(input("Enter sepal length: "))
  b = float(input("Enter sepal width: "))
  c = float(input("Enter petal length: "))
  d = float(input("Enter petal width: "))

  pred = model.predict(np.array([[a, b, c, d]], np.float64))
  result = le.inverseTransform(pred)

  print("It's a ", result[0])
