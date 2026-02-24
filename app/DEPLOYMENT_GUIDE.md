# Face Mask Detector - Complete Setup Guide

## 🚀 Quick Start (LOCALHOST - Development)

### Step 1: Install Dependencies
```powershell
cd D:\Projects\face_mask_detector\app
pip install -r requirements.txt
```

### Step 2: Run Locally
```powershell
python app.py
```

**Expected Output:**
```
🚀 Running in DEVELOPMENT mode (HTTP only)
   Access at: http://localhost:7860
```

### Step 3: Test in Browser
```
✅ http://localhost:7860  (Camera works - HTTP is fine on localhost!)
```

---

## 📋 Testing Checklist (Localhost)

- [ ] App starts without errors
- [ ] Visit http://localhost:7860
- [ ] Photo upload works
- [ ] Camera capture works (click "Start Camera")
- [ ] Video upload works
- [ ] Live camera feed works (if camera connected)
- [ ] Results display correctly

---

## 🌐 Production Deployment

Once testing is complete on localhost, choose your deployment option:

### **Option 1: Railway.app (⭐ RECOMMENDED - Easiest)**

Railway automatically handles HTTPS and deployment.

**Step 1: Sign up**
```
https://railway.app
```

**Step 2: Connect GitHub**
- GitHub → Railway → Select repo
- Railway auto-deploys on git push

**Step 3: Set Environment Variables in Railway Dashboard**
- `FLASK_ENV` = `production`
- `PORT` = `7860`
- `DEBUG` = `False`

**Step 4: Deploy**
```powershell
git push origin main
# Railway auto-deploys - takes 2-3 minutes
```

**Access:** `https://yourapp.railway.app` ✅

---

### **Option 2: Heroku (Free Tier Ending - Alternative)**

**Step 1: Install Heroku CLI**
```powershell
# Download from https://devcenter.heroku.com/articles/heroku-cli
heroku login
```

**Step 2: Create Procfile**
Create `Procfile` (no extension) in project root:
```
web: gunicorn --bind 0.0.0.0:$PORT wsgi:app --timeout 120
```

**Step 3: Deploy**
```powershell
heroku create your-app-name
heroku config:set FLASK_ENV=production
git push heroku main
```

**Step 4: View Logs**
```powershell
heroku logs --tail
```

---

### **Option 3: Docker + Any Cloud (AWS, GCP, Azure, DigitalOcean)**

#### **Build Docker Image Locally**
```powershell
docker build -t face-mask-detector .
docker run -p 7860:7860 \
  -e FLASK_ENV=production \
  face-mask-detector
```

#### **Deploy to AWS EC2**
```bash
# 1. Create EC2 instance (Ubuntu 20.04)
# 2. SSH into instance
ssh -i your-key.pem ec2-user@your-ip

# 3. Install Docker
sudo yum update -y
sudo yum install docker -y
sudo systemctl start docker

# 4. Push image to ECR (AWS Container Registry)
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin YOUR-ECR-URI
docker tag face-mask-detector:latest YOUR-ECR-URI/face-mask-detector:latest
docker push YOUR-ECR-URI/face-mask-detector:latest

# 5. Run container
docker run -d -p 443:7860 \
  -e FLASK_ENV=production \
  YOUR-ECR-URI/face-mask-detector:latest
```

#### **Setup HTTPS with Let's Encrypt (Free)**
```bash
sudo apt update
sudo apt install certbot python3-certbot-nginx -y

# Get certificate for yourdomain.com
sudo certbot certonly --standalone -d yourdomain.com

# Configure Nginx as reverse proxy
sudo apt install nginx -y
# Edit /etc/nginx/sites-available/default with SSL config
sudo systemctl restart nginx
```

---

### **Option 4: Google Cloud Run**

```powershell
# 1. Install Google Cloud CLI
# 2. Authenticate
gcloud auth login
gcloud config set project YOUR-PROJECT-ID

# 3. Build and push to Container Registry
gcloud builds submit --tag gcr.io/YOUR-PROJECT-ID/face-mask-detector

# 4. Deploy to Cloud Run
gcloud run deploy face-mask-detector \
  --image gcr.io/YOUR-PROJECT-ID/face-mask-detector \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars FLASK_ENV=production
```

---

### **Option 5: DigitalOcean Droplet + GitHub Actions (CI/CD)**

#### **Step 1: Create Droplet**
- Ubuntu 20.04, 2GB RAM
- Create SSH key

#### **Step 2: Setup Server**
```bash
ssh root@your-droplet-ip

# Update system
apt update && apt upgrade -y

# Install dependencies
apt install python3-pip python3-venv docker.io git -y

# Clone repo
git clone https://github.com/yourusername/face-mask-detector.git
cd face-mask-detector/app

# Setup virtual environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### **Step 3: Setup Gunicorn as Service**
Create `/etc/systemd/system/face-mask.service`:
```ini
[Unit]
Description=Face Mask Detector
After=network.target

[Service]
User=root
WorkingDirectory=/root/face-mask-detector/app
Environment="FLASK_ENV=production"
ExecStart=/root/face-mask-detector/app/venv/bin/gunicorn --bind 0.0.0.0:5000 wsgi:app --timeout 120
Restart=always

[Install]
WantedBy=multi-user.target
```

Start service:
```bash
systemctl enable face-mask
systemctl start face-mask
systemctl status face-mask
```

#### **Step 4: Setup Nginx + SSL**
```bash
apt install nginx certbot python3-certbot-nginx -y

# Get SSL certificate
certbot certonly --nginx -d yourdomain.com

# Create Nginx config
# Edit /etc/nginx/sites-available/default
```

Nginx config example:
```nginx
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
        proxy_read_timeout 120s;
    }

    # Static files caching
    location /static/ {
        expires 30d;
        add_header Cache-Control "public, immutable";
    }
}

# Redirect HTTP to HTTPS
server {
    listen 80;
    server_name yourdomain.com;
    return 301 https://$server_name$request_uri;
}
```

Enable and restart:
```bash
nginx -t
systemctl restart nginx
```

#### **Step 5: Setup GitHub Actions for Auto-Deploy**
Create `.github/workflows/deploy.yml`:
```yaml
name: Deploy to DigitalOcean

on:
  push:
    branches: [ main ]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Deploy to server
        uses: appleboy/ssh-action@master
        with:
          host: ${{ secrets.SERVER_IP }}
          username: root
          key: ${{ secrets.SSH_KEY }}
          script: |
            cd face-mask-detector
            git pull origin main
            cd app
            source venv/bin/activate
            pip install -r requirements.txt
            systemctl restart face-mask
```

Add GitHub Secrets:
- `SERVER_IP`: Your DigitalOcean droplet IP
- `SSH_KEY`: Your private SSH key

---

## 🔄 Local Development Workflow

### **Terminal 1: Run Flask App**
```powershell
cd D:\Projects\face_mask_detector\app
python app.py
```

### **Terminal 2: Watch for Changes (Optional)**
```powershell
# Install Flask reload
pip install python-dotenv flask-cors

# App auto-reloads when files change (debug=True enabled)
```

### **Browser**
```
http://localhost:7860
```

---

## 📦 Environment Variables

### **Development (.env)**
```
FLASK_ENV=development
PORT=7860
DEBUG=True
```

### **Production (.env.production)**
```
FLASK_ENV=production
PORT=7860
DEBUG=False
```

---

## 🔒 Security Checklist

### **Development Mode (Localhost)**
- ✅ HTTP (fine for local testing)
- ✅ Debug mode enabled (auto-reload)
- ✅ No authentication needed

### **Production Mode (Cloud)**
- ✅ HTTPS (mandatory)
- ✅ Debug=False
- ✅ SSL certificate (Let's Encrypt or paid)
- ✅ Input validation
- ✅ Rate limiting
- ✅ CSRF protection
- ✅ Secure file upload handling
- ✅ Regular backups

---

## 🐛 Troubleshooting

### **Localhost won't open**
```powershell
# Make sure .env has:
# FLASK_ENV=development
# and cert.pem/key.pem are NOT being used

# Restart app
python app.py
```

### **Camera not working on localhost**
```
✅ Localhost HTTP is fine - Camera API works!
❌ Only non-localhost HTTP fails (use HTTPS instead)
```

### **Port 7860 already in use**
```powershell
# Find what's using the port
Get-NetTCPConnection -LocalPort 7860 | Select-Object OwningProcess
# Kill process
Stop-Process -Id <PID> -Force
```

### **ModuleNotFoundError on deploy**
```
✅ Check requirements.txt has all packages
✅ Redeploy/restart container
✅ Check deployment logs
```

---

## 📊 Performance Tips

1. **Enable Caching**
   ```python
   from flask_caching import Cache
   cache = Cache(app, config={'CACHE_TYPE': 'redis'})
   ```

2. **Use CDN for Static Files**
   - CSS/JS via CloudFlare CDN
   - Save bandwidth

3. **Database Optimization**
   - Use Redis for caching
   - Connection pooling

4. **Model Optimization**
   - Use model quantization
   - TensorFlow Lite for mobile

---

## 📈 Monitoring & Logging

### **Local Logs**
```bash
tail -f app.log
```

### **Cloud Monitoring**
- **Railway**: Built-in dashboard
- **Heroku**: `heroku logs --tail`
- **Docker**: `docker logs container-id`
- **DigitalOcean**: SystemD logs & app logs

### **Add Logging to App**
```python
import logging
logging.basicConfig(filename='app.log', level=logging.INFO)
app.logger.info('Event occurred')
```

---

## 💡 Next Steps

1. ✅ Test on localhost (http://localhost:7860)
2. ✅ Commit to GitHub
3. ✅ Choose deployment option (Railway recommended)
4. ✅ Deploy to production
5. ✅ Setup domain name
6. ✅ Monitor & maintain

---

## 📞 Support

- **Framework**: Flask https://flask.palletsprojects.com/
- **Deployment**: Railway https://railway.app
- **SSL Certificates**: Let's Encrypt https://letsencrypt.org/
- **Docker**: https://www.docker.com/

