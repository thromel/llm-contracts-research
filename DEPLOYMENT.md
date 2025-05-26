# Deployment Guide

## 🚀 Production Deployment Checklist

### Prerequisites
- [ ] Python 3.8+ installed
- [ ] Git repository cloned
- [ ] Dependencies installed (`requirements.txt` + `requirements-agentic.txt`)

### External Services Setup

#### 1. MongoDB Atlas
- [ ] Create MongoDB Atlas account
- [ ] Create new cluster (free tier sufficient for testing)
- [ ] Create database user with read/write permissions
- [ ] Whitelist IP addresses (or use 0.0.0.0/0 for testing)
- [ ] Get connection string
- [ ] Set `MONGODB_URI` environment variable

#### 2. API Keys
- [ ] **OpenAI API Key** (for GPT-4.1 borderline screening)
  - [ ] Create account at https://platform.openai.com
  - [ ] Generate API key
  - [ ] Set `OPENAI_API_KEY` environment variable
  
- [ ] **DeepSeek API Key** (for bulk screening)
  - [ ] Create account at https://platform.deepseek.com
  - [ ] Generate API key
  - [ ] Set `DEEPSEEK_API_KEY` environment variable

#### 3. Data Sources (Optional)
- [ ] **GitHub Token** (for data acquisition)
  - [ ] Generate personal access token
  - [ ] Set `GITHUB_TOKEN` environment variable
  
- [ ] **Stack Overflow API** (for data acquisition)
  - [ ] Register application
  - [ ] Set API credentials

### Environment Configuration

Create `.env` file or set environment variables:

```bash
# Required for production mode
MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/llm_contracts_research
OPENAI_API_KEY=sk-proj-...
DEEPSEEK_API_KEY=sk-...

# Optional
GITHUB_TOKEN=ghp_...
SCREENING_MODE=traditional
MAX_POSTS_PER_RUN=1000
```

### Testing Checklist

#### 1. Component Tests
```bash
python test_simple_pipeline.py
```
Expected: ✅ All 6 tests pass

#### 2. Mock Mode Test
```bash
python run_simple_screening.py --mock
```
Expected: ✅ 100% success rate, 3 posts processed

#### 3. Production Mode Test (with real services)
```bash
python run_simple_screening.py --max-posts 5
```
Expected: ✅ Successful connection to MongoDB and APIs

### Deployment Steps

1. **Clone and Setup**
   ```bash
   git clone <repository-url>
   cd llm-contracts-research
   pip install -r requirements.txt
   pip install -r requirements-agentic.txt
   ```

2. **Configure Environment**
   ```bash
   export MONGODB_URI="your-mongodb-uri"
   export OPENAI_API_KEY="your-openai-key"
   export DEEPSEEK_API_KEY="your-deepseek-key"
   ```

3. **Validate Setup**
   ```bash
   python test_simple_pipeline.py
   python run_simple_screening.py --mock
   ```

4. **Production Test**
   ```bash
   python run_simple_screening.py --max-posts 10
   ```

## 🔧 Troubleshooting

### Common Issues

#### 1. Import Errors
**Problem**: `ModuleNotFoundError` when running scripts
**Solution**: 
- Ensure you're in the project root directory
- Check that all dependencies are installed
- Verify Python path includes the project directory

#### 2. MongoDB Connection Failed
**Problem**: `ConnectionFailure` or timeout errors
**Solution**:
- Verify `MONGODB_URI` is correct
- Check network connectivity
- Ensure IP is whitelisted in MongoDB Atlas
- Verify database user has correct permissions

#### 3. API Key Issues
**Problem**: `401 Unauthorized` or `403 Forbidden` errors
**Solution**:
- Verify API keys are correct and not expired
- Check API usage limits and quotas
- Ensure proper environment variable names
- Test API keys directly with curl/Postman

#### 4. Rate Limiting
**Problem**: `429 Too Many Requests` errors
**Solution**:
- Reduce batch sizes in configuration
- Add delays between API calls
- Upgrade API plan for higher limits
- Use multiple API keys for rotation

### Performance Optimization

#### For High Volume Processing
1. **MongoDB Optimization**
   - Use MongoDB Atlas M10+ for better performance
   - Enable connection pooling
   - Add indexes for frequently queried fields

2. **API Optimization**
   - Use concurrent processing within rate limits
   - Implement exponential backoff for retries
   - Cache common results

3. **Memory Management**
   - Process data in batches
   - Clear intermediate results
   - Monitor memory usage

### Monitoring

#### Key Metrics to Track
- Processing rate (posts/minute)
- API response times
- Error rates by component
- Database query performance
- Memory/CPU usage

#### Logs to Monitor
- Component initialization status
- API connection health
- Processing statistics
- Error messages and stack traces

### Scaling Considerations

#### For Research-Scale Deployment (10K+ posts)
1. **Horizontal Scaling**
   - Multiple worker processes
   - Load balancing across instances
   - Distributed task queues

2. **Database Scaling**
   - MongoDB sharding
   - Read replicas for analytics
   - Optimized indexes

3. **API Management**
   - Multiple API key rotation
   - Request caching
   - Batch processing optimization

## 📊 Expected Performance

### Mock Mode
- **Setup Time**: < 1 second
- **Processing Rate**: 1000+ posts/second
- **Memory Usage**: < 100MB

### Production Mode
- **Setup Time**: 2-5 seconds (database connection)
- **Processing Rate**: 10-50 posts/minute (API limited)
- **Memory Usage**: 200-500MB

### Typical Batch Processing
- **100 posts**: 3-10 minutes
- **1,000 posts**: 30-60 minutes  
- **10,000 posts**: 5-10 hours

## 🆘 Support

If you encounter issues not covered here:

1. **Check Logs**: Review detailed error messages
2. **Test Components**: Run individual component tests
3. **Mock Mode**: Verify system works in mock mode
4. **Documentation**: Check `docs/` for detailed guides
5. **Issues**: Create GitHub issue with:
   - Error messages/stack traces
   - Environment details (Python version, OS)
   - Steps to reproduce
   - Configuration (anonymized)

## ✅ Production Readiness Checklist

- [ ] All component tests pass
- [ ] Mock mode works correctly
- [ ] External services configured and tested
- [ ] Environment variables set properly
- [ ] Monitoring and logging configured
- [ ] Backup and recovery plan in place
- [ ] Performance tested with expected load
- [ ] Error handling validated
- [ ] Documentation updated
- [ ] Team trained on operations

---

**Next Steps**: Once deployment is complete, proceed with data acquisition and begin processing real posts! 