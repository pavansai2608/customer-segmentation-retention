pipeline {
    agent any

    environment {
        DAGSHUB_CREDS = credentials('dagshub-token') // Jenkins credential ID (user + token)
    }

    stages {
        stage('Backend: install & test') {
            agent {
                docker { image 'python:3.11-slim' }
            }
            steps {
                sh 'pip install -r requirements.txt dvc'
                sh '''
                    dvc remote modify origin --local auth basic
                    dvc remote modify origin --local user "$DAGSHUB_CREDS_USR"
                    dvc remote modify origin --local password "$DAGSHUB_CREDS_PSW"
                    dvc pull || echo "DVC pull failed/skipped — model-dependent tests may fail"
                '''
                dir('backend') {
                    sh 'pytest -v'
                }
            }
        }

        stage('Frontend: install, test & build') {
            agent {
                docker { image 'node:20-alpine' }
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
}