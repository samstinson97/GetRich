#!/bin/bash
# One-time AWS setup for signal-gen Lambda + EventBridge.
# Prerequisite: AWS CLI configured (aws configure), with admin or scoped IAM access.
# Run from project root: bash production/cloud/setup_aws.sh

set -e

REGION=us-east-1
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_REPO=getrich-signal-gen
LAMBDA_FN=getrich-signal-gen
STATE_BUCKET=getrich-runtime-state-$ACCOUNT_ID
ROLE_NAME=getrich-signal-gen-role
EVENTBRIDGE_RULE=getrich-signal-gen-daily

echo "=== Account: $ACCOUNT_ID  Region: $REGION ==="
echo "  ECR repo:        $ECR_REPO"
echo "  Lambda fn:       $LAMBDA_FN"
echo "  State bucket:    $STATE_BUCKET"
echo "  IAM role:        $ROLE_NAME"
echo "  EventBridge:     $EVENTBRIDGE_RULE"
echo ""

# 1. ECR repository
echo "[1/6] Creating ECR repo..."
aws ecr describe-repositories --repository-names $ECR_REPO --region $REGION 2>/dev/null \
  || aws ecr create-repository --repository-name $ECR_REPO --region $REGION

# 2. S3 state bucket
echo "[2/6] Creating S3 state bucket..."
aws s3api head-bucket --bucket $STATE_BUCKET 2>/dev/null \
  || aws s3api create-bucket --bucket $STATE_BUCKET --region $REGION

# 3. Lambda execution role with S3 access
echo "[3/6] Creating IAM role $ROLE_NAME..."
TRUST_DOC='{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": {"Service": "lambda.amazonaws.com"},
    "Action": "sts:AssumeRole"
  }]
}'

if ! aws iam get-role --role-name $ROLE_NAME 2>/dev/null; then
  aws iam create-role --role-name $ROLE_NAME --assume-role-policy-document "$TRUST_DOC"
  aws iam attach-role-policy --role-name $ROLE_NAME \
    --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole

  S3_POLICY=$(cat <<EOF
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Action": ["s3:GetObject", "s3:PutObject", "s3:ListBucket"],
    "Resource": ["arn:aws:s3:::$STATE_BUCKET", "arn:aws:s3:::$STATE_BUCKET/*"]
  }]
}
EOF
  )
  aws iam put-role-policy --role-name $ROLE_NAME \
    --policy-name s3-state-rw --policy-document "$S3_POLICY"
  echo "  Waiting 10s for IAM role propagation..."
  sleep 10
fi
ROLE_ARN=$(aws iam get-role --role-name $ROLE_NAME --query Role.Arn --output text)
echo "  Role ARN: $ROLE_ARN"

# 4. Build + push initial image (so Lambda has something to point at)
echo "[4/6] Building initial Docker image..."
aws ecr get-login-password --region $REGION | \
  docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com
docker build -f production/cloud/Dockerfile -t $ECR_REPO:initial .
docker tag $ECR_REPO:initial $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$ECR_REPO:initial
docker tag $ECR_REPO:initial $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$ECR_REPO:latest
docker push $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$ECR_REPO:initial
docker push $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$ECR_REPO:latest

# 5. Lambda function (container image)
echo "[5/6] Creating Lambda function $LAMBDA_FN..."
IMAGE_URI=$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$ECR_REPO:latest

if ! aws lambda get-function --function-name $LAMBDA_FN --region $REGION 2>/dev/null; then
  aws lambda create-function \
    --function-name $LAMBDA_FN \
    --package-type Image \
    --code ImageUri=$IMAGE_URI \
    --role $ROLE_ARN \
    --timeout 900 \
    --memory-size 2048 \
    --environment "Variables={STATE_S3_BUCKET=$STATE_BUCKET}" \
    --region $REGION
  aws lambda wait function-active --function-name $LAMBDA_FN --region $REGION
else
  aws lambda update-function-code \
    --function-name $LAMBDA_FN \
    --image-uri $IMAGE_URI \
    --region $REGION
  aws lambda wait function-updated --function-name $LAMBDA_FN --region $REGION
fi

# 6. EventBridge rule for 3:55 PM ET (19:55 UTC during EDT, 20:55 UTC during EST)
# NOTE: AWS cron is in UTC. To handle daylight saving, this uses 19:55 UTC year-round
# (= 3:55 PM EDT during DST, 2:55 PM EST during standard time).
# CRITICAL: This will be 1 hour EARLY during standard time (Nov-Mar). Adjust manually.
# Better: use two rules with different schedules and disable/enable via Lambda layer.
echo "[6/6] Creating EventBridge rule (cron 55 19 ? * MON-FRI *)..."
aws events put-rule \
  --name $EVENTBRIDGE_RULE \
  --schedule-expression "cron(55 19 ? * MON-FRI *)" \
  --description "Trigger getrich signal-gen at 19:55 UTC (3:55 PM EDT) Mon-Fri" \
  --region $REGION

# Lambda permission for EventBridge to invoke
aws lambda add-permission \
  --function-name $LAMBDA_FN \
  --statement-id eventbridge-invoke \
  --action lambda:InvokeFunction \
  --principal events.amazonaws.com \
  --source-arn arn:aws:events:$REGION:$ACCOUNT_ID:rule/$EVENTBRIDGE_RULE \
  --region $REGION 2>/dev/null || true

# Connect rule -> Lambda
aws events put-targets \
  --rule $EVENTBRIDGE_RULE \
  --targets "Id=1,Arn=arn:aws:lambda:$REGION:$ACCOUNT_ID:function:$LAMBDA_FN" \
  --region $REGION

echo ""
echo "=== DONE ==="
echo "Lambda fn:        $LAMBDA_FN"
echo "EventBridge rule: $EVENTBRIDGE_RULE (cron 55 19 ? * MON-FRI *)"
echo "State bucket:     s3://$STATE_BUCKET/runtime-state/"
echo ""
echo "Next steps:"
echo "  1. Set Lambda env vars for Alpaca API keys (one per account):"
echo "     V3_FRESH_API_KEY, V3_FRESH_API_SECRET"
echo "     V4_GTC_API_KEY,   V4_GTC_API_SECRET"
echo "     V4_MOC_API_KEY,   V4_MOC_API_SECRET"
echo "  2. Upload current state files to S3:"
echo "     aws s3 cp production/ml_v3/state_fresh.json s3://$STATE_BUCKET/runtime-state/"
echo "     aws s3 cp production/ml_v4/state_gtc.json   s3://$STATE_BUCKET/runtime-state/"
echo "     aws s3 cp production/ml_v4/state_moc.json   s3://$STATE_BUCKET/runtime-state/"
echo "  3. Test invoke: aws lambda invoke --function-name $LAMBDA_FN /tmp/out.json --region $REGION"
echo "  4. Disable Windows Task Scheduler 'GetRich_AllSignals'"
