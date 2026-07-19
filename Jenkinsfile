pipeline {
    agent any

    environment {
        DAGSHUB_CREDS = credentials('dagshub-token')
        // Absolute path to your real, already-populated local checkout on this Mac.
        // This is where models/ and data/ actually have real file content (pulled via
        // `dvc pull` once, manually, outside of CI). We mount those directories straight
        // into the container so tests don't depend on a network dvc pull succeeding
        // during every single CI run.
        LOCAL_REPO_WITH_DATA = "${HOME}/Desktop/customer-segmentation-retention"
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
pip install -r requirements.txt
cd backend
pytest -v --junitxml=test-results.xml
EOF
                    chmod +x run_backend_tests.sh

                    docker run --rm -u root \
                        -v "$WORKSPACE":/workspace \
                        -v "$LOCAL_REPO_WITH_DATA/models":/workspace/models \
                        -v "$LOCAL_REPO_WITH_DATA/data":/workspace/data \
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