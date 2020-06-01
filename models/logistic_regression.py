import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def run_code():
	print ('entering')
	train_data = pd.read_csv('../data/train.csv')
	test_data = pd.read_csv('../data/test.csv')

	x = train_data.drop('label', axis=1)
	y = train_data['label']
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)

	linear = LogisticRegression(
		C=20, solver='lbfgs', multi_class='multinomial'
	)
	linear.fit(x_train, y_train)
	print('fitting done')
	y_hat = linear.predict(x_test)
	score = accuracy_score(y_test.values, y_hat.round())
	print('score', score)

	x_final_test = test_data.drop("id", axis=1)
	y_final_test = linear.predict(x_final_test)

	submission = pd.DataFrame({'id': test_data.id, 'label': y_final_test})
	submission['label'] = submission['label'].astype('int')
	submission.to_csv('../data/submission.csv', index=False)

if __name__ == '__main__':
	run_code()