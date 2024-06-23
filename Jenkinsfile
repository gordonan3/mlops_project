pipeline {
    agent any

    stages {
        stage('Checkout') {
            steps {
                git branch: 'main', url: 'https://github.com/gordonan3/mlops_project.git'
            }
        }
        stage('Install Dependencies') {
            steps {
                sh 'pip3 install pandas scikit-learn numpy joblib'
            }
        }
        stage('Fetch Data') {
            steps {
                sh 'python3 data_creation.py'
            }
        }
        stage('Preprocess Data') {
            steps {
                sh 'python3 data_preprocessing.py'
            }
        }
        stage('Fetch Dataset') {
            steps {
                sh 'python3 dataset_creation.py'
            }
        }
        stage('Testing Dataset') {
            steps {
                sh 'python3 dataset_test.py'
            }
        }
        stage('Train Model') {
            steps {
                sh 'python3 model_preparation.py'
            }
        }
    }
    
}
