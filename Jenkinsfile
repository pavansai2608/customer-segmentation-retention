pipeline {
    agent any

    environment {
        DAGSHUB_CREDS = credentials('dagshub-token')
    }

    options {
        timestamps()
        disableConcurrentBuilds()
    }

    stages {
        stage('Backend: install & test') {
            agent {
                docker { image 'python:3.12-slim'; args '-u root' }
            }
            steps {
                sh 'pip install -r requirements.txt dvc dvc-http httpx'
                sh '''
                    dvc remote modify origin --local auth basic
                    dvc remote modify origin --local user "$DAGSHUB_CREDS_USR"
                    dvc remote modify origin --local password "$DAGSHUB_CREDS_PSW"
                    dvc pull || echo "DVC pull failed/skipped — model-dependent tests may fail"
                '''
                dir('backend') {
                    sh 'pytest -v --junitxml=test-results.xml'
                }
            }
            post {
                always {
                    junit 'backend/test-results.xml'
                }
            }
        }

        stage('Frontend: install, test & build') {
            agent {
                docker { image 'node:20-alpine' }
            }
            environment {
                REACT_APP_API_URL = 'http://127.0.0.1:8000'
            }
            steps {
                dir('frontend/my-app') {
                    sh 'npm ci'
                    sh 'CI=true npm test -- --watchAll=false'
                    sh 'npm run build'
                }
            }
        }
    }

    post {
        success { echo 'Pipeline passed.' }
        failure { echo 'Pipeline failed — check stage logs above.' }
    }
}