import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC


def run_code():
	print('entering')
	train_data = pd.read_csv('../data/train.csv')
	test_data = pd.read_csv('../data/test.csv')

	x = train_data.drop('label', axis=1)
	y = train_data['label']
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15)

	steps = [('pca', PCA()), ('svm', SVC())]
	pipeline = Pipeline(steps)

	parameters = {
		'svm__C': [9, 0.9],
		'pca__n_components': [0.8, 0.7],
		'pca__whiten': [True]
	}

	grid = GridSearchCV(pipeline, param_grid=parameters, cv=5, n_jobs=-1)
	grid.fit(x_train, y_train)
	print("score = %3.2f" % (grid.score(x_test, y_test)))
	print(grid.best_params_)


if __name__ == '__main__':
	run_code()
