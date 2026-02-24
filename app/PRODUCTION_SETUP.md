# Production Deployment Guide

## Current Status
✅ HTTPS certificate generated (cert.pem, key.pem)
✅ Flask app configured for HTTPS
✅ Docker support added

---

## Quick Start (Development)
```powershell
# Ensure dependencies installed
pip install -r requirements.txt

# Start with HTTPS
python app.py

# Visit: https://192.168.1.144:7860
# (Accept self-signed certificate warning)
```

---

## Production Deployment Options

### **Option A: Heroku (Free Tier)**
```bash
# Install Heroku CLI
# Create Procfile:
web: gunicorn --certfile=cert.pem --keyfile=key.pem --bind 0.0.0.0:$PORT wsgi:app

# Deploy
heroku create your-app-name
git push heroku main
```

### **Option B: AWS EC2 with Nginx**
```bash
# 1. Get real SSL certificate from Certbot/Let's Encrypt
sudo certbot certonly --standalone -d yourdomain.com

# 2. Configure Nginx
server {
    listen 443 ssl http2;
    server_name yourdomain.com;
    
    ssl_certificate /etc/letsencrypt/live/yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/yourdomain.com/privkey.pem;
    
    location / {
        proxy_pass http://localhost:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}

# 3. Run Python with Gunicorn
gunicorn --bind 0.0.0.0:5000 wsgi:app
```

### **Option C: Docker (Any Cloud)**
```bash
# Build
docker build -t face-mask-detector .

# Run
docker run -p 443:443 \
  -v $(pwd)/cert.pem:/app/cert.pem \
  -v $(pwd)/key.pem:/app/key.pem \
  face-mask-detector

# Visit: https://your-server-ip
```

### **Option D: Google Cloud Run**
```bash
# Build and push to Google Container Registry
gcloud builds submit --tag gcr.io/your-project/face-mask-detector

# Deploy
gcloud run deploy face-mask-detector \
  --image gcr.io/your-project/face-mask-detector \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

### **Option E: Railway.app (Recommended for Starters)**
```bash
# Railway automatically handles HTTPS
# 1. Connect GitHub repo
# 2. Set PORT=7860 environment variable
# 3. Deploy
```

---

## SSL Certificate Options

### **Self-Signed (Development/Internal)**
```bash
# Already created! Just use cert.pem and key.pem
# Valid for 365 days
```

### **Free SSL (Production)**
```bash
# Use Let's Encrypt with Certbot
sudo apt-get install certbot python3-certbot-nginx

# Get certificate for your domain
sudo certbot certonly --nginx -d yourdomain.com -d www.yourdomain.com

# Renews automatically every 90 days
```

### **Paid SSL (Optional)**
- Comodo, DigiCert, RapidSSL
- More trust indicators in browsers
- Better for enterprise/e-commerce

---

## Environment Variables (Production)

Create `.env` file:
```
FLASK_ENV=production
DEBUG=False
MAX_CONTENT_LENGTH=500
GPU_ENABLED=True  # if available
```

Load in app.py:
```python
from dotenv import load_dotenv
import os

load_dotenv()
DEBUG = os.getenv('DEBUG', 'False') == 'True'
app.run(debug=DEBUG)
```

---

## Security Checklist

- [ ] Using HTTPS (not HTTP)
- [ ] Valid SSL certificate (not self-signed for production)
- [ ] DEBUG=False in production
- [ ] Set strong SECRET_KEY
- [ ] Use environment variables for secrets
- [ ] Rate limiting enabled
- [ ] File upload validation
- [ ] CORS properly configured
- [ ] Regular certificate renewal (Let's Encrypt)

---

## Testing HTTPS Locally

```powershell
# Start Flask with HTTPS
python app.py

# Test in browser
# ✅ https://192.168.1.144:7860 (works)
# ❌ http://192.168.1.144:7860 (camera won't work)
# ✅ localhost works on HTTP (special case)
```

---

## Performance Tips

1. **Enable Caching**
   ```python
   from flask_caching import Cache
   cache = Cache(app, config={'CACHE_TYPE': 'simple'})
   ```

2. **Use CDN for Static Files**
   - CSS/JS should be served from CDN
   - Models can be cached

3. **Enable Gzip Compression**
   ```bash
   # Nginx
   gzip on;
   gzip_types text/css application/javascript;
   ```

4. **Database Optimization**
   - Use Redis for caching
   - Connection pooling
   - Query optimization

---

## Monitoring & Logging

```python
import logging
from logging.handlers import RotatingFileHandler

handler = RotatingFileHandler('app.log', maxBytes=10000000, backupCount=5)
handler.setLevel(logging.INFO)
app.logger.addHandler(handler)
```

---

## Troubleshooting

**Camera still not working?**
- ✅ Check HTTPS is enabled (`https://` in URL bar)
- ✅ Certificate is valid (not expired)
- ✅ Not using HTTP origin

**Deployment fails on cloud?**
- Ensure cert.pem & key.pem in git repo (or mount as volumes)
- Check PORT environment variable
- Verify all dependencies in requirements.txt

