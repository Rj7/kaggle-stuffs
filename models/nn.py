import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split


def run_evaluation():

	train_data = pd.read_csv('../data/train.csv')
	test_data = pd.read_csv('../data/test.csv')
	dig_data = pd.read_csv("../data/Dig-MNIST.csv")

	x = train_data.drop('label', axis=1)
	y = train_data['label']
	x = x.values
	y = y.values

	x_dig = dig_data.drop('label', axis=1)
	y_dig = dig_data['label']
	x_dig = x_dig.values
	y_dig = y_dig.values
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)

	model = tf.keras.models.Sequential([
		tf.keras.layers.Dense(128, activation='relu'),
		tf.keras.layers.Dropout(0.2),
		tf.keras.layers.Dense(10, activation='softmax')
	])

	model.compile(
		optimizer='adam',
		loss='sparse_categorical_crossentropy',
		metrics=['accuracy']
	)

	model.fit(x_train, y_train, epochs=20)

	model.evaluate(x_test, y_test, verbose=2)
	model.evaluate(x_dig, y_dig)


if __name__ == '__main__':
	run_evaluation()
