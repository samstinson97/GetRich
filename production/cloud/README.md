# Cloud deployment — AWS Lambda container

Daily signal generator (3:55 PM ET weekdays) running serverless on AWS so the local Windows machine can be off.

## Architecture

```
EventBridge cron (19:55 UTC Mon-Fri)
      ↓ triggers
AWS Lambda (container, 2GB mem, 15min timeout)
      ↓ at start
S3 (runtime-state/) → /tmp/state/  (state.json, ohlcv_cache.pkl)
      ↓ runs
v3_fresh signal → v4_gtc signal → v4_moc signal → dashboard → reconcile
      ↓ at end
/tmp/state/ → S3 (runtime-state/)
```

Container image stored in ECR. Updated via GitHub Actions on push to main.

## Cost

~$0.30/mo total:
- Lambda compute (5 runs × 5 min × 2GB): ~$0.05/mo (within 400K GB-sec free tier)
- ECR image storage (3GB): ~$0.30/mo
- EventBridge: free
- S3 state: ~$0.01/mo

## Files

| File | Purpose |
|------|---------|
| `Dockerfile` | Container image for Lambda |
| `requirements.txt` | Python deps |
| `lambda_handler.py` | Entry point — calls v3/v4 run_signal() in sequence |
| `s3_state.py` | Sync state.json + ohlcv_cache.pkl to/from S3 |
| `setup_aws.sh` | One-time AWS infra creation (ECR, Lambda, IAM, EventBridge, S3) |
| `../../.github/workflows/deploy-lambda.yml` | CI: rebuild + push image on commit |

## First-time setup

### 1. Get a private GitHub repo

Create `getrich` (or similar) under your GitHub account. Push the project:

```bash
cd /c/ai-research-team
git init
git add production/ v10_ml_v3/seed_6/ v10_ml_v4_sortino/saved_model_1188_s6/ \
        v10_2_engine.py production/cloud/ .github/workflows/
git commit -m "Initial commit"
git remote add origin git@github.com:<you>/getrich.git
git push -u origin main
```

**WARNING**: ensure `.gitignore` excludes `production/accounts/*.env`, `*.pkl`, `state*.json` — these contain secrets/state.

### 2. Run the AWS setup script

```bash
# Configure AWS CLI first if not already:
aws configure
# (paste access key, secret, region us-east-1)

bash production/cloud/setup_aws.sh
```

This creates: ECR repo, S3 bucket, IAM role, Lambda function (with initial image), EventBridge rule.

### 3. Set Lambda env vars for Alpaca keys

```bash
# Read keys from local .env files
V3_KEY=$(grep ALPACA_API_KEY production/accounts/v3_fresh.env | cut -d= -f2)
V3_SECRET=$(grep ALPACA_SECRET_KEY production/accounts/v3_fresh.env | cut -d= -f2)
V4G_KEY=$(grep ALPACA_API_KEY production/accounts/v4_gtc.env | cut -d= -f2)
V4G_SECRET=$(grep ALPACA_SECRET_KEY production/accounts/v4_gtc.env | cut -d= -f2)
V4M_KEY=$(grep ALPACA_API_KEY production/accounts/v4_moc.env | cut -d= -f2)
V4M_SECRET=$(grep ALPACA_SECRET_KEY production/accounts/v4_moc.env | cut -d= -f2)

# Update Lambda environment
aws lambda update-function-configuration \
  --function-name getrich-signal-gen \
  --environment "Variables={STATE_S3_BUCKET=getrich-runtime-state-$(aws sts get-caller-identity --query Account --output text),V3_FRESH_API_KEY=$V3_KEY,V3_FRESH_API_SECRET=$V3_SECRET,V4_GTC_API_KEY=$V4G_KEY,V4_GTC_API_SECRET=$V4G_SECRET,V4_MOC_API_KEY=$V4M_KEY,V4_MOC_API_SECRET=$V4M_SECRET}" \
  --region us-east-1
```

### 4. Upload current state to S3

```bash
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
BUCKET=getrich-runtime-state-$ACCOUNT

aws s3 cp production/ml_v3/state_fresh.json     s3://$BUCKET/runtime-state/state_fresh.json
aws s3 cp production/ml_v3/ohlcv_cache.pkl      s3://$BUCKET/runtime-state/ohlcv_cache_v3.pkl
aws s3 cp production/ml_v4/state_gtc.json       s3://$BUCKET/runtime-state/state_gtc.json
aws s3 cp production/ml_v4/state_moc.json       s3://$BUCKET/runtime-state/state_moc.json
aws s3 cp production/ml_v4/ohlcv_cache_gtc.pkl  s3://$BUCKET/runtime-state/ohlcv_cache_gtc.pkl
aws s3 cp production/ml_v4/ohlcv_cache_moc.pkl  s3://$BUCKET/runtime-state/ohlcv_cache_moc.pkl
```

### 5. Test invoke

```bash
aws lambda invoke \
  --function-name getrich-signal-gen \
  --payload '{}' /tmp/lambda-out.json \
  --region us-east-1
cat /tmp/lambda-out.json
```

If status="ok", the Lambda is working.

### 6. (After 1-2 successful test invocations) disable the Windows Task Scheduler

Open Task Scheduler → find `GetRich_AllSignals` → Disable.

## GitHub Actions setup (optional, for auto-deploy on push)

1. Create an IAM role for GitHub OIDC. AWS docs:
   https://docs.aws.amazon.com/IAM/latest/UserGuide/id_roles_create_for-idp_oidc.html

2. Add repo secret: `AWS_DEPLOY_ROLE_ARN` = the role ARN.

3. Push to main. Workflow will auto-rebuild and deploy.

## Daylight saving caveat

EventBridge cron uses UTC. The rule `cron(55 19 ? * MON-FRI *)` fires at:
- 3:55 PM EDT during Daylight Saving Time (Mar-Nov)
- 2:55 PM EST during Standard Time (Nov-Mar) — **1 hour early!**

To handle this manually, twice a year update the rule:
```bash
# When DST ends (Nov):
aws events put-rule --name getrich-signal-gen-daily \
  --schedule-expression "cron(55 20 ? * MON-FRI *)" --region us-east-1
# When DST begins (Mar):
aws events put-rule --name getrich-signal-gen-daily \
  --schedule-expression "cron(55 19 ? * MON-FRI *)" --region us-east-1
```

Or use two rules + a Lambda layer that picks based on date. Out of scope for v1.

## Manual invoke / debugging

```bash
# Tail logs:
aws logs tail /aws/lambda/getrich-signal-gen --region us-east-1 --follow

# Re-invoke immediately:
aws lambda invoke --function-name getrich-signal-gen /tmp/out.json --region us-east-1

# Check S3 state freshness:
aws s3 ls s3://getrich-runtime-state-$(aws sts get-caller-identity --query Account --output text)/runtime-state/
```
