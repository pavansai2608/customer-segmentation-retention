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
            steps {
                sh '''
                    cat > run_backend_tests.sh << 'EOF'
#!/bin/sh
set -e
pip install -r requirements.txt dvc dvc-http httpx
dvc remote modify origin --local auth basic
dvc remote modify origin --local user "$DAGSHUB_CREDS_USR"
dvc remote modify origin --local password "$DAGSHUB_CREDS_PSW"
dvc pull || echo "DVC pull failed/skipped — model-dependent tests may fail"
cd backend
pytest -v --junitxml=test-results.xml
EOF
                    chmod +x run_backend_tests.sh

                    docker run --rm -u root \
                        -e DAGSHUB_CREDS_USR \
                        -e DAGSHUB_CREDS_PSW \
                        -v "$WORKSPACE":/workspace \
                        -w /workspace \
                        python:3.12-slim ./run_backend_tests.sh
                '''
            }
            post {
                always {
                    junit 'backend/test-results.xml'
                }
            }
        }

        stage('Frontend: install, test & build') {
            environment {
                REACT_APP_API_URL = 'http://127.0.0.1:8000'
            }
            steps {
                sh '''
                    cat > run_frontend.sh << 'EOF'
#!/bin/sh
set -e
cd frontend/my-app
npm ci
CI=true npm test -- --watchAll=false
npm run build
EOF
                    chmod +x run_frontend.sh

                    docker run --rm \
                        -e REACT_APP_API_URL \
                        -v "$WORKSPACE":/workspace \
                        -w /workspace \
                        node:20-alpine ./run_frontend.sh
                '''
            }
        }
    }

    post {
        always {
            sh 'rm -f run_backend_tests.sh run_frontend.sh'
        }
        success { echo 'Pipeline passed.' }
        failure { echo 'Pipeline failed — check stage logs above.' }
    }
}