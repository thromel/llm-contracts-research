# GitHub Secrets Setup Guide

This guide explains how to set up GitHub repository secrets for the LLM Contract Analysis Dashboard deployment.

## Required Secrets

### 1. MONGODB_URI
Your MongoDB connection string.

**Examples:**
```bash
# Local MongoDB
mongodb://localhost:27017

# MongoDB Atlas
mongodb+srv://username:password@cluster.mongodb.net/

# MongoDB with authentication
mongodb://username:password@localhost:27017/admin
```

### 2. DATABASE_NAME
Your database name (optional, defaults to `llm_contracts_research`).

**Example:**
```
llm_contracts_research
```

### 3. OPENAI_API_KEY (Optional)
Your OpenAI API key for live analysis (if needed).

**Example:**
```
sk-proj-...your-key-here...
```

## How to Add Secrets

### Step 1: Go to Repository Settings
1. Navigate to your GitHub repository
2. Click **Settings** tab
3. In the left sidebar, click **Secrets and variables**
4. Click **Actions**

### Step 2: Add Each Secret
1. Click **New repository secret**
2. Enter the secret name (e.g., `MONGODB_URI`)
3. Enter the secret value
4. Click **Add secret**

### Step 3: Verify Secrets
After adding, you should see:
- ✅ `MONGODB_URI`
- ✅ `DATABASE_NAME` (optional)
- ✅ `OPENAI_API_KEY` (optional)

## Environment Setup

### For Production Environment
If you want to use different secrets for production:

1. Go to **Settings** → **Environments**
2. Click **New environment**
3. Name it `production`
4. Add environment-specific secrets

### Environment Variables in Workflow
The enhanced workflow will automatically:
- Try to connect to MongoDB using your secrets
- Export live data if connection succeeds
- Fall back to sample data if connection fails
- Display the data source in the dashboard

## Security Best Practices

### ✅ DO:
- Use MongoDB Atlas with restricted IP access
- Create read-only database users for CI/CD
- Use strong passwords
- Rotate secrets regularly
- Use private repositories for sensitive data

### ❌ DON'T:
- Commit secrets to code
- Use production credentials for testing
- Share secrets outside your team
- Use admin privileges for dashboard access

## Connection String Examples

### MongoDB Atlas (Recommended)
```bash
MONGODB_URI=mongodb+srv://dashboard-user:SecurePassword123@your-cluster.mongodb.net/?retryWrites=true&w=majority
DATABASE_NAME=llm_contracts_research
```

### Local Development
```bash
MONGODB_URI=mongodb://localhost:27017
DATABASE_NAME=llm_contracts_research
```

### Docker Compose
```bash
MONGODB_URI=mongodb://mongodb:27017
DATABASE_NAME=llm_contracts_research
```

### With Authentication
```bash
MONGODB_URI=mongodb://username:password@host:port/admin?authSource=admin
DATABASE_NAME=llm_contracts_research
```

## Testing Connection

### Local Testing
```bash
# Set environment variables
export MONGODB_URI="your-connection-string"
export DATABASE_NAME="llm_contracts_research"

# Test connection
python -c "
import os
import pymongo

try:
    client = pymongo.MongoClient(os.getenv('MONGODB_URI'))
    client.server_info()
    print('✅ Connection successful')
    
    db = client[os.getenv('DATABASE_NAME', 'llm_contracts_research')]
    collections = db.list_collection_names()
    print(f'📁 Collections: {collections}')
    
except Exception as e:
    print(f'❌ Connection failed: {e}')
"
```

### GitHub Actions Testing
The workflow will automatically test the connection and show results in the Action logs.

## Deployment Workflow

### Automatic Deployment
The enhanced workflow (`deploy-dashboard-enhanced.yml`) will:

1. **Test MongoDB Connection**
   - Try to connect using secrets
   - Check for required collections
   - Report connection status

2. **Export Data**
   - If connected: Export live data from collections
   - If not connected: Generate sample data

3. **Build Dashboard**
   - Include appropriate data file
   - Set data source indicator
   - Deploy to GitHub Pages

### Manual Deployment
You can trigger deployment manually:

1. Go to **Actions** tab
2. Select "Deploy Enhanced Streamlit Dashboard"
3. Click **Run workflow**
4. Choose whether to use live data
5. Click **Run workflow**

## Troubleshooting

### Connection Fails
```
❌ MongoDB connection failed: [Errno 11001] getaddrinfo failed
```
**Solution:** Check MONGODB_URI format and network access

### Authentication Fails
```
❌ Authentication failed
```
**Solution:** Verify username/password in connection string

### Database Not Found
```
❌ Database 'llm_contracts_research' not found
```
**Solution:** Check DATABASE_NAME or create database

### No Collections
```
📁 Collections: []
```
**Solution:** Run the pipeline first to populate data

### Workflow Fails
1. Check **Actions** tab for error logs
2. Verify all required secrets are set
3. Test connection string locally
4. Check repository permissions

## Advanced Configuration

### Custom Collection Names
If your collections have different names, modify the workflow:

```python
# In the workflow, change:
'labelled_posts' → 'your_collection_name'
```

### Read-Only User
Create a dedicated user for the dashboard:

```javascript
// In MongoDB shell
use admin
db.createUser({
  user: "dashboard-reader",
  pwd: "SecurePassword123",
  roles: [
    { role: "read", db: "llm_contracts_research" }
  ]
})
```

### Connection Pooling
For better performance, use connection options:

```bash
MONGODB_URI=mongodb+srv://user:pass@cluster.net/?retryWrites=true&w=majority&maxPoolSize=10
```

## Support

If you encounter issues:
1. Check the GitHub Actions logs
2. Test connection locally first
3. Verify secret values (re-add if necessary)
4. Check MongoDB server status
5. Review network/firewall settings